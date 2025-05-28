git reset --hard
rm -rf smartcash
git add .
git commit -am "main update"
git push origin main --force
git checkout migration