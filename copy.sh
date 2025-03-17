cp -r smartcash/model .
cp -r smartcash/configs .
cp -r smartcash/components .
cp -r smartcash/ui .
cp -r smartcash/dataset .
cp -r smartcash/detection .
cp -r smartcash/common .
git add .
git commit -am "update"
git push origin migration
git rm -rf ui