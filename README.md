# Pose Embeddings Explore

## Installation on Windows 

**Step 1:** Install Visual Studio 2017 [here](https://download.visualstudio.microsoft.com/download/pr/4035d2dd-2d45-48eb-9104-d4dc7d808a7b/f5675416a31cbf8c29e74d75a1790cf7/vs_community.exe)


**Step 1:** Install Anaconda distribtion [here](https://www.anaconda.com/download)


**Step 2:** Install git 
```
conda install -c anaconda git 
```

**Step 3:** Clone the repository
```
git clone https://github.com/marybethcassity/PoseEmbeddingsExplore.git
```

**Step 4:** Navigate into the directory
```
cd /path/to/PoseEmbeddingsExplore
```

**Step 5:** Create the anaconda environment 
```
conda env create --name embeddings -f requirements.yml 
```
(you can replace embeddings with any other environment name you want)


**Step 6:** Activate the environment 
```
conda activate embeddings 
```
(make sure to replace embeddings with your environemnt name if you came up with your own)


**Step 7:** Run the app 

```
python main.py
```

## Installation on WSL for GPU Enablement
