python3 -m mypy --html-report=mypy-report src
coverage run -m unittests tests/*
coverage report -m --omit="tests/*"
coverage html --omit="tests/*"
xdg-open htmlcov/index.html