import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    imgs = sorted(os.listdir(inputdir))
    for idx,img in enumerate(imgs):
        groups = ''

        groups += os.path.join(inputdir, img) + '|'
        groups += os.path.join(args.mask, img) + '|'
        groups += os.path.join(targetdir,img)

        # if idx >= 800:
        #     break

        with open(os.path.join(outputdir, 'groups_test_ICCV.txt'), 'a') as f:
            f.write(groups + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/jieh/Projects/Shadow/Dataset/Dataset1/test/input', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--mask', type=str, default='/media/jieh/Backup Plus/HuangAbla/mask/crf_ICCV21', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--target', type=str, default='/home/jieh/Projects/Shadow/Dataset/Dataset1/test/gt', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/home/jieh/Projects/Shadow/MainNet/data/', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()
