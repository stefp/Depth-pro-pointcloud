# Depth-pro-pointcloud
This repo includes a post-processing step following Apple's [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073) to produce a point cloud from the output depth map.

![image](https://github.com/user-attachments/assets/839a835b-55ec-417a-8413-c67d7b585f01)

🚀 First Iteration Alert! 🚀
Help is needed to ensure that the code outputs a point cloud in the metric system. Your expertise is welcomed!

# 🛠️ Setup 
Clone the original [Depth-pro repo](https://github.com/apple/ml-depth-pro) repo and follow the instructions to setup the environment

 ```
git clone https://github.com/apple/ml-depth-pro
```

Clone this repo

 ```
git clone https://github.com/stefp/Depth-pro-pointcloud
cd Depth-pro-pointcloud
```

# ▶️ Run 
Run the following
 ```
python depth-pro-pointcloud.py C:\Users\stpu\Downloads\forest.jfif
```

💡 Tip: You can change the image (*.jpeg, *.jpg, *.png, etc.) to any image you have on hand!

# 🙏 Help Needed!
It seems the output scale might be off. Anyone willing to help fix this? 🤔

