# Welcome to your Jupyter Book

This is a small sample book to give you a feel for how book content is
structured.

:::{note}
Here is a note!
:::

And here is a code block:

```
e = mc^2
```

Check out the content pages bundled with this sample book to see more.


In the terminal type 

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



ghp-import pushes collections of HTMLL files onto the "gh-pages" braanch of a GitHub repo

push only the ghp page to GitHub 