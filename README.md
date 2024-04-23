# Pose Embeddings Explore

## Installation on Windows 

**Step 1:** Install Visual Studio 2017 [here](https://download.visualstudio.microsoft.com/download/pr/4035d2dd-2d45-48eb-9104-d4dc7d808a7b/f5675416a31cbf8c29e74d75a1790cf7/vs_community.exe)

Make sure to install Desktop development with C++  

![image](https://github.com/marybethcassity/PoseEmbeddingsExplore/assets/70182553/30b20a59-4fbb-418b-b735-5dea12b8bfef)


**Step 2:** Restart your computer


**Step 3:** Install Anaconda distribtion [here](https://www.anaconda.com/download)


**Step 4:** Install git 
```
conda install -c anaconda git 
```

**Step 5:** Clone the repository
```
git clone https://github.com/marybethcassity/PoseEmbeddingsExplore.git
```

**Step 6:** Navigate into the directory
```
cd /path/to/PoseEmbeddingsExplore
```

**Step 7:** Create the anaconda environment 
```
conda env create --name embeddings -f requirements.yml 
```
(you can replace embeddings with any other environment name you want)


**Step 8:** Activate the environment 
```
conda activate embeddings 
```
(make sure to replace embeddings with your environemnt name if you came up with your own)


**Step 9:** Run the app 

```
python main.py
```

## Installation on WSL for GPU Enablement
