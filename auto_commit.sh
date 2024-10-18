
autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r src
isort src
git add .
git commit -m "[modify] reformat the code for release."
git push 