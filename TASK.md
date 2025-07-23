Updated 22 Juli 2025, 20:
# NOTES:
- Raw Data: `/content/data/{valid, train, test}/{images, labels}` or `/data/{valid, train, test}/{images, labels} (Symlink|Local)`
- Preprocessed Data: `/content/data/preprocessed/{valid, train, test}/{images, labels}` or `/data/preprocessed/{valid, train, test}/{images, labels} (Symlink|Local)`
- Augmented Data: `/content/data/augmented/{valid, train, test}/{images, labels}` or `/data/augmented/{valid, train, test}/{images, labels} (Symlink|Local)`
- Pretrained Data: `/content/datata/pretrained` or `/data/pretrained (Symlink|Local)`
- Checkpoint Data: `/content/data/checkpoints` or `/data/checkpoints (Symlink|Local)`
- Model Data: `/content/data/models` or `/data/models (Symlink|Local)`

# GENERAL
- Create Consistent Summary Format for Downloader, Preprocessing and Augmentation UI Module

# COLAB MODULE
- ✅ All Works

# DEPENDENCY MODULE
- [ ] Install, Uninstall and updates process use synchronous 
- [ ] Check missing packages and updates should use threadpool for faster response
- [ ] Ensure no overlapping operations. 

# DOWNLOAD MODULE
- [ ] Each operation done should update summary result 

# PREPROCESSING MODULE
- [ ] Create quick backend service readiness check. Add new api on `smartcash/dataset/preprocessor/api` that do quick check (empty or not) on `/dataset/preprocessed/{train, valid, test}/{labels, images}`
- [ ] Add post init hook using quick check api and set service ready flag in preprocessing ui module
- [ ] Do not run download and cleanup operation if service not ready
- [ ] If service ready, open confirmation dialog if existing data found
- [ ] Operation summary missing new line formatting. 

# SPLIT MODULE
- ✅ All Works

# AUGMENTATION MODULE
- [ ] Summary Container on UI dictionary but not yet updated when operations done. Should be like preprocessing operation summary. 

# PRETRAINED MODULE

# BACKBONE MODULE
- Dynamic Button Registration still poluting operation logs when it should run once on init. It happend during clicking action buttons.


# TRAINING MODULE

# EVALUATION MODULE

# VISUALIZATION MODULE