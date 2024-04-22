import os
import glob as gb



data_path = '/path/to/dataset/'
rgb_path = data_path + '/JPEGImages'  
# '/JPEGImages/480p' for DAVIS-related datasets and '/JPEGImages' for others

gap = [1]
reverse = [0, 1]
batch_size = 4

folder = gb.glob(os.path.join(rgb_path, '*'))
for r in reverse:
    for g in gap:
        for f in folder:
            print('===> Runing {}, gap {}'.format(f, g))
            mode = 'raft-things.pth'  # model
            if r==1:
                raw_outroot = data_path + '/Flows_gap-{}/'.format(g)  # where to raw flow
                outroot = data_path + '/FlowImages_gap-{}/'.format(g)  # where to save the image flow
            elif r==0:
                raw_outroot = data_path + '/Flows_gap{}/'.format(g)   # where to raw flow
                outroot = data_path + '/FlowImages_gap{}/'.format(g)   # where to save the image flow
                
            os.system("python predict.py "
                        "--gap {} --mode {} --path {} --batch_size {} "
                        "--outroot {} --reverse {} --raw_outroot {}".format(g, mode, f, batch_size, outroot, r, raw_outroot))

      
      
