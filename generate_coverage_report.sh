coverage run -m unittests tests/*.py
coverage report
coverage html
xdg-open htmlcov/index.html