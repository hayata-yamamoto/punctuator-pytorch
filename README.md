# punctuator
Data Science Project. Have Fun!

# Get Started 

```bash
$ python3 -m venv venv 
$ . venv/bin/activate
$ python3 setup.py install # or develop
```


# Structure 
```markdown
punctuator-pytorch/
├── LICENSE
├── README.md
├── data
│   ├── interim
│   ├── processed
│   └── raw
├── punctuator
│   ├── __init__.py
│   ├── manage.py
│   ├── modules
│   ├── notebooks
│   │   ├── exploratory
│   │   └── predictive
│   ├── src
│   │   ├── __init__.py
│   │   ├── core
│   │   │   ├── __init__.py
│   │   │   ├── commands.py
│   │   │   ├── path_manager.py
│   │   │   └── settings.py
│   │   ├── datasets
│   │   │   └── __init__.py
│   │   ├── features
│   │   │   └── __init__.py
│   │   └── models
│   │       └── __init__.py
│   └── tests
│       └── __init__.py
├── requirements.txt
└── setup.py
```
