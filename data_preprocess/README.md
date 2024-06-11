## 数据集制作流水线
1. 手机录制一段视频video.mp4
2. 选一帧无表情的，用MICA推理，得到identity.npy
3. 输入视频和identity.npy，用metrical tracker推理，得到相机参数（checkpoint文件夹）
4. 下载预处理的images文件夹中的图片，合成一个视频（因为metrical tracker帧数会有问题）
5. 用IMAvatar的流水线推理，得到flame_params.json，images和mask
6. 将checkpoint文件夹和flame_params.json放到本文件夹下，依次调用metrical.py & merge.py得到transforms.json
7. images命名为ori_imgs，和mask & transforms.json打包为最终数据集