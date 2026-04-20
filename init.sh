python -m venv myenv
source ~/image_ai/myenv/bin/activate
pip install --upgrade pip
pip install tensorflow[and-cuda]==2.20.0                # Installs TF with GPU support (includes CUDA runtime)
pip install keras numpy pandas matplotlib scikit-learn  # Extras if needed
pip install seaborn
pip install pathlib
pip install pillow>=9.0.0
pip install scipy>=1.9.0
pip install tqdm>=4.65.0
pip install pyyaml>=6.0
deactivate