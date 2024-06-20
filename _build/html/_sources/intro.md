# Machine Learning in Python



___
This is a quick note on how to create a Jupyter Book in Jupyter Lab.

```
conda activate jb

pip installl -U jupyter-book
conda installl -c conda-forge jupyter-book

jb --help
jb create mybookname
jb build mybookname
```
make sure when you create and build book, you are not in the `mybookname` folder. 


```
git clone https://github.com/mikenguyen13/mlpy
```


then to push to github, make sure you are inside the `mybookname` folder using `cd mybookname`. Then, 

```
pip install ghp-import
ghp-import -n -p -f _build/html/
```

ghp-import pushes collections of HTMLL files onto the "gh-pages" branch of a GitHub repo

push only the ghp page to GitHub 