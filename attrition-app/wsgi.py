"""
wsgi.py
-------
WSGI entry point used by PythonAnywhere (and any other WSGI-compatible host).

PythonAnywhere config example (Web tab -> WSGI configuration file):

    import sys
    project_home = '/home/<your-username>/employee-attrition-prediction'
    if project_home not in sys.path:
        sys.path.insert(0, project_home)

    from wsgi import application

Gunicorn / Render / Railway do not need this file — they run `app:app`
directly per the Procfile — but it's kept here for completeness and for
any traditional WSGI host that expects an `application` callable.
"""

from app import app as application

if __name__ == "__main__":
    application.run()
