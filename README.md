# pyDIC
[![DOI](https://zenodo.org/badge/64115684.svg)](https://zenodo.org/badge/latestdoi/64115684)

A Digital Image Correlation application, developed in Python.

## Getting Started

This package is hosted on pypi. To install it, simply run:

```
$ python -m pip install py_dic
```

You can now start the GUI by running:

```
$ python -m py_dic
```

### Running the source code 

Clone or download the contetnts into a local directory. Install the package via `setupy.py` or run the `main.py` file and follow the on-screen instructions. 

```
$ python -m py_dic.main
```

To change the default analysis settings, edit the `settings.ini` file.

You can use the `run.bat` file to start the GUI on Windows, but it expects the Python interpreter to be located in the `./venv/Scripts` folder inside the project's root directory. To achieve this, run the following commands (on a Windows machine):

```
$ cd pyIDC
$ python -m pip virtualenv venv
$ venv\Scripts\activate.bat

```

to create a ner virtual environment inside the `venv` folder and activate is. Now, install the project requirements into this activated virtual environment:

```
$ python -m pip install -r requirements.txt
```

Now, you can execute the `run.bat` file to start the pyDIC GUI.


### Authors

- [Domen Gorjup](http://ladisk.si/?what=incfl&flnm=gorjup.php)
- [Janko Slavič](http://ladisk.si/?what=incfl&flnm=slavic.php)
- [Miha Boltežar](http://ladisk.si/?what=incfl&flnm=boltezar.php)
