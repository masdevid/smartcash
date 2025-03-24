rm -rf model
rm -rf configs
rm -rf components
rm -rf ui
rm -rf dataset
rm -rf detection
rm -rf common
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
rm -rf ui
rm -rf model
rm -rf configs
rm -rf components
rm -rf dataset
rm -rf detection
rm -rf common