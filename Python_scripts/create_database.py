'''
Copyright (c) 2019 Netflix, Inc., University of Texas at Austin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authors:
Somdyuti Paul <somdyuti@utexas.edu>
Andrey Norkin <anorkin@netflix.com>
Alan C. Bovik <bovik@ece.utexas.edu>
'''

import subprocess
import numpy as np
import os
import argparse
import math


def run_process(cmd, **kwargs):
    ret = subprocess.call(cmd, **kwargs)
    assert ret == 0, 'Process returned {ret}, cmd: {cmd}'.format(ret=ret, cmd=cmd)
    return ret

def get_resolution(ffprobe_path, encode):
    get_height = """{ffprobe} -loglevel panic -hide_banner -v error -show_entries stream=height -of default=noprint_wrappers=1 {encode} | grep -oP \"(?<=height\=)[0-9]+\"""".format(ffprobe=ffprobe_path, encode=encode)
    get_width = """{ffprobe} -hide_banner -v error -show_entries stream=width -of default=noprint_wrappers=1 {encode} | grep -oP \"(?<=width\=)[0-9]+\"""".format(
        ffprobe=ffprobe_path, encode=encode)
    height = subprocess.check_output(get_height, shell=True)
    width = subprocess.check_output(get_width, shell=True)
    return height,width


def get_kf_numbers(ffprobe_path, input):

    get_kf_cmd = """{ffprobe} -loglevel panic -hide_banner -select_streams v -show_frames -show_entries frame=pict_type -of csv {input} | grep -n I | cut -d ':' -f 1""".format(
            input=input,
            ffprobe=ffprobe_path,
            )

    output = subprocess.check_output(get_kf_cmd, shell=True)
    return output

def downsamplencrop_source(ffmpeg_path, input_path, output_path, width, height, kf_indices,key):
    kfs = """eq(n\, {num})""".format(num=kf_indices[0])
    for i in range(1, len(kf_indices)):
        kfs = kfs + """+eq(n\, {num})""".format(num=kf_indices[i])
    crops = ""
    num_blks = 0
    n = (int)(math.floor(float(width) / 64) * math.floor(float(height) / 64))
    split = str(n)
    maps=""
    for y in range(0, height, 64):
        for x in range(0, width, 64):
            if (x + 64 <= width and y + 64 <= height):
                num_blks += 1
                if(n==num_blks):
                    crops = crops + """[in{numblks}]crop=66:66:{x}:{y}[out{numblks}]""".format(numblks=num_blks, x=x, y=y)
                else:
                    crops = crops + """[in{numblks}]crop=66:66:{x}:{y}[out{numblks}];""".format(numblks=num_blks, x=x, y=y)
                split = split + """[in{numblks}]""".format(numblks=num_blks)
                maps = maps + """-c:v rawvideo -pix_fmt yuv420p -map [out{numblks}] {path}/{key}_kf_%d_{numblks}.png """.format(path=output_path, numblks=num_blks, key=key)

    cmd = """{ffmpeg} -loglevel panic -hide_banner -y -i {input_path} -sws_flags lanczos+accurate_rnd -vsync 0 -filter_complex  "[0:v]select='""".format(ffmpeg=ffmpeg_path, input_path=input_path)+kfs+"""', scale={width}:{height}, split=""".format(width=width, height=height)+split+ """;"""+crops+ """\" """+maps
    run_process(cmd, shell=True)

def write_merge_info(partition,blk_size,q, blk_no, kf_no, output_path,key):
    if('-' in partition):
        return
    else:
        partition=list(map(int,partition))
        partition = np.asarray(partition)

        blk_size=list(map(int,blk_size))
        blk_size=np.asarray(blk_size)

        # Level 0 (64x64 block, output 1X1 merge map)
        L0 = np.zeros((1,1),int)
        L1 = np.zeros((2,2),int)

        ind1 =np.zeros((2,2),int)


        # 0 - Full Merge
        # 3 - No Merge

        # Level 0 - merge info for the 4 largest 32x32 blocks is directly given by root of tree
        L0[0,0]=partition[0]

        # Level 1 - if L0 = 0,1 or 2 there should be full merge of 16x16 blocks
        if(L0[0,0]!=3):
            L1.fill(0)
            ind1.fill(0)
        else:
            ind=np.argwhere(blk_size == 9)
            L1[0,0]=partition[ind[0]]
            L1[0,1]=partition[ind[1]]
            L1[1,0] = partition[ind[2]]
            L1[1,1] = partition[ind[3]]

            ind1[0,0]=ind[0]
            ind1[0,1] = ind[1]
            ind1[1,0] = ind[2]
            ind1[1,1] = ind[3]

        # Level 2: Say for L1[0,0]: If L1[0,0]=0,1,2, the corresponding subblocks will have full merge. if L1[0,0]=3, look for 1st 4 occurences of block size 6
        k=0
        l=0
        count=0
        L2_list = []
        L2 = np.zeros((4, 4), int)
        I2 = np.zeros((4, 4), int)
        for i in range(2):
            for j in range(2):
                if (L1[i, j] != 3):
                    L2_piece = np.zeros((2, 2), int)
                    I2_piece = np.zeros((2, 2), int)
                else:
                    if(count!=3):
                        piece_blk_size = blk_size[ind1[i, j] + 1:ind1[np.unravel_index(count+1,(2,2))]]
                        piece_partition=partition[ind1[i, j] + 1:ind1[np.unravel_index(count+1,(2,2))]]
                    else:
                        piece_blk_size = blk_size[ind1[i, j]+1:]
                        piece_partition = partition[ind1[i, j]+1:]


                    ind = np.argwhere(piece_blk_size == 6)
                    L2_piece = np.empty((2, 2), int)
                    I2_piece = np.empty((2, 2), int)
                    L2_piece[0, 0] = piece_partition[ind[0]]
                    L2_piece[0, 1] = piece_partition[ind[1]]
                    L2_piece[1, 0] = piece_partition[ind[2]]
                    L2_piece[1, 1] = piece_partition[ind[3]]

                    I2_piece[0,0]=ind[0]+ind1[i,j]+1
                    I2_piece[0, 1] = ind[1]+ind1[i,j]+1
                    I2_piece[1, 0] = ind[2]+ind1[i,j]+1
                    I2_piece[1, 1] = ind[3]+ind1[i,j]+1


                L2[k:k+2,l:l+2]=L2_piece
                I2[k:k+2,l:l+2]=I2_piece
                L2_list.append(L2_piece)
                l=l+2
                count += 1


            l=0
            k=k+2

        # Level 3: Proceed with same logic as level 2.

        L3 = np.zeros((8,8),int)
        L3_half = np.zeros((4,4),int)

        k=0
        l=0
        I2_sorted = np.trim_zeros(np.sort(I2, axis=None))

        for i in range(4):
            for j in range(4):
                if (L2[i, j] != 3):
                    L3_piece = np.zeros((2, 2), int)
                else:
                    next=np.argwhere(I2_sorted==I2[i,j])+1
                    if(next<len(I2_sorted)):
                        piece_blk_size = blk_size[I2[i, j] + 1:int(I2_sorted[next])]
                        piece_partition = partition[I2[i, j] + 1:int(I2_sorted[next])]
                    else:
                        piece_blk_size = blk_size[I2[i, j] + 1:]
                        piece_partition = partition[I2[i, j] + 1:]


                    ind = np.argwhere(piece_blk_size == 3)
                    L3_piece = np.empty((2, 2), int)
                    L3_piece[0, 0] = piece_partition[ind[0]]
                    L3_piece[0, 1] = piece_partition[ind[1]]
                    L3_piece[1, 0] = piece_partition[ind[2]]
                    L3_piece[1, 1] = piece_partition[ind[3]]

                L3[k:k + 2, l:l + 2] = L3_piece
                l += 2

            l = 0
            k = k + 2
    directory = output_path + "/"+ key+ "_kf_" + str(kf_no+1)
    op_file=directory+"_"+str(blk_no)
    q=int(q)
    np.savez(op_file, q=q,L0=L0, L1=L1,L2=L2,L3=L3)



parser = argparse.ArgumentParser()
parser.add_argument('--ffmpeg_path')
parser.add_argument('--ffprobe_path')
parser.add_argument('--source_path')
parser.add_argument('--encode_path')
parser.add_argument('--stats_path')
parser.add_argument('--output_path')
parser.add_argument('--key')
args = parser.parse_args()

source = args.source_path
encode = args.encode_path
stats = args.stats_path
ffmpeg = args.ffmpeg_path
ffprobe= args.ffprobe_path
output=args.output_path
key=args.key

height, width=get_resolution(ffprobe,encode)
H = int(height)
W = int(width)

kf_indices=get_kf_numbers(ffprobe, encode)
kf_indices=np.fromstring(kf_indices, dtype=int, sep=' ')

path = args.output_path
if not os.path.exists(path):
        os.makedirs(path)

# start = time.time()
downsamplencrop_source(ffmpeg,source,path,W,H,kf_indices,key)
# end = time.time()

w=0
h=0
blkno=0
valid_blk=[]
for y in range(0, H, 64):
    for x in range(0, W, 64):
        if(x+64<=W and y+64<=H):
            blkno+=1
            valid_blk.append(1)
        else:
            valid_blk.append(0)

kf_no=["Frame #"+ str(s) for s in kf_indices]
frame_count=0
with open(stats) as statsFile:
    for line_no, line in enumerate(statsFile):
        if line.startswith("Frame #"):
            frame_no = line


        if line.startswith("Frame Type 0"):

           if frame_no.strip() not in kf_no:
               raise ValueError("Wrong stats file")

           for t in range(3):
                line=next(statsFile)
           blk_cnt = 1
           partitions_arr=[]
           blk_size_arr=[]
           q=0
           while(not line.startswith("Frame #")):
                # print(frame_no)
                info=line.split('\t')
                if(info[0].strip().isdigit()):
                    if(partitions_arr and q):
                       if(valid_blk[blk_cnt-1]==1):
                           write_merge_info(partitions_arr,blk_size_arr, q,blk_cnt,frame_count,path,key)
                           blk_cnt = blk_cnt + 1
                    partitions_arr=[]
                    blk_size_arr = []
                    q=0
                    partitions_arr.append(info[4].strip())
                    blk_size_arr.append(info[2].strip())
                    q=info[6].strip()
                elif(info[0].strip()=='-' or info[0] == ' ' or info[0].strip=="\t"):
                    partitions_arr.append(info[4].strip())
                    blk_size_arr.append(info[2].strip())
                line = next(statsFile, None)
                if(line==None):
                    break
           frame_count=frame_count+1
