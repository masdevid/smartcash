# rm -rf model
# rm -rf configs
# rm -rf components
# rm -rf ui
# rm -rf dataset
# rm -rf detection
# rm -rf common
echo "Start..."
cp -r smartcash/model .
cp -r smartcash/configs .
cp -r smartcash/components .
cp -r smartcash/ui .
cp -r smartcash/dataset .
cp -r smartcash/detection .
cp -r smartcash/common .
sleep 1
git add .
git commit -am "update"
git push origin migration
sleep 3
rm -rf ui
rm -rf model
rm -rf configs
rm -rf components
rm -rf dataset
rm -rf detection
rm -rf common
echo "Done!"