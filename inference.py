import os
import glob as gb
import argparse
import cv2
import os
import requests

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Variable to keep track of frame count
    frame_count = 0
    
    # Read frames until there are no more
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame as an image
        frame_path = os.path.join(output_folder, f"{frame_count+1:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1

    # Release the video capture object
    cap.release()

def extract_flow(rgb_path, flow_output_path):
    gap = [1]
    reverse = [0, 1]
    batch_size = 4

    folder = gb.glob(os.path.join(rgb_path, '*'))
    for r in reverse:
        for g in gap:
            for f in folder:
                print('===> Running {}, gap {}'.format(f, g))
                mode = 'flow/raft-things.pth'  # model
                if r==1:
                    raw_outroot = flow_output_path + '/Flows_gap-{}/'.format(g)  # where to raw flow
                    outroot = flow_output_path + '/FlowImages_gap-{}/'.format(g)  # where to save the image flow
                elif r==0:
                    raw_outroot = flow_output_path + '/Flows_gap{}/'.format(g)   # where to raw flow
                    outroot = flow_output_path + '/FlowImages_gap{}/'.format(g)   # where to save the image flow
                    
                os.system("python flow/predict.py "
                            "--gap {} --mode {} --path {} --batch_size {} "
                            "--outroot {} --reverse {} --raw_outroot {}".format(g, mode, f, batch_size, outroot, r, raw_outroot))

def create_video_from_images(input_folder, output_video_path, fps):
    # Get the list of image files in the input folder
    image_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.png')]

    # Sort the image files by name
    image_files.sort()

    # Get the dimensions of the first image to set the video size
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use appropriate codec based on the output video format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate over each image and add it to the video
    for image_file in image_files:
        img = cv2.imread(image_file)
        out.write(img)

    # Release VideoWriter object
    out.release()

def download_weight(filename, url):
    # Check if the file exists locally
    if not os.path.exists(filename):
        print(f"File '{filename}' not found locally. Proceeding with download.")

        # Download the file
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

        print("File downloaded successfully!")
    else:
        print(f"File '{filename}' already exists locally. No need to download.")

def inference(args):
    """
    User should change the configuration path to appropriate path.
    Install segment-anything
    """
    ckpt = 'frame_level_flowpsam_vitbvith_train_on_oclrsyn_dvs17m.pth'
    rgb_encoder_ckpt_path = 'sam_vit_h_4b8939.pth'
    download_weight(rgb_encoder_ckpt_path, 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')
    flow_encoder_ckpt_path = 'sam_vit_b_01ec64.pth'
    download_weight(flow_encoder_ckpt_path, 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth')
    os.system("python evaluation.py "
                "--model flowpsam --ckpt {} --rgb_encoder_ckpt_path {} --flow_encoder_ckpt_path {} --flow_gaps 1 "
                "--dataset example --save_path {}".format(ckpt, rgb_encoder_ckpt_path, flow_encoder_ckpt_path, args.flowsam_output_path))

"""
python inference.py --video_file_path sample.mp4 --video_output_path output/images/sample --extract_frames --flow_output_path output/flow --extract_flow --visualize_flow --run_flowsam --flowsam_output_path output --visualize_output
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file_path', type=str, help="restore checkpoint")
    parser.add_argument('--video_output_path', type=str, help="restore checkpoint")
    parser.add_argument('--extract_frames', action='store_true', help='convert video file to image file folder')

    parser.add_argument('--flow_output_path', type=str, help="restore checkpoint")
    parser.add_argument('--extract_flow', action='store_true', help='whether to run flow ')
    parser.add_argument('--visualize_flow', action='store_true', help='whether to run flow ')

    parser.add_argument('--flowsam_output_path', type=str, help="restore checkpoint")
    parser.add_argument('--run_flowsam', action='store_true', help='whether to run flow ')
    parser.add_argument('--visualize_output', action='store_true', help='whether to run flow ')
    args = parser.parse_args()

    if args.extract_frames:
        extract_frames(args.video_file_path, args.video_output_path)

    # Split the path into directory and filename
    directory, filename = os.path.split(args.video_output_path)

    if args.extract_flow:
        extract_flow(directory, args.flow_output_path)
    # (Optional) For debug and visualization purpose
    if args.visualize_flow:
        flow_path = os.path.join(args.flow_output_path, f'FlowImages_gap-1/{filename}')
        create_video_from_images(flow_path, f'{args.flow_output_path}/flow.mp4', fps=30)

    if args.run_flowsam:
        inference(args=args)

    if args.visualize_output:
        output_path = os.path.join(args.flowsam_output_path, f"nonhung/{filename}")
        create_video_from_images(output_path, 'output.mp4', fps=30)