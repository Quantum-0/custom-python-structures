coverage run -m unittests tests/*
coverage report
coverage html
xdg-open htmlcov/index.html