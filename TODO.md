# TODO

- [ ] Update `src/data/split_data.py`:
  - [ ] Ensure split uses correct input under `dataset/processed`
  - [ ] Keep class folder structure in `dataset/split/{train,val,test}/{class}`
  - [ ] Reset `SPLIT_DIR` before copying (remove old split)
  - [ ] Remove duplicates inside processed input (MD5-bytes) before splitting
  - [ ] Copy without basename collisions
- [ ] Run `python src/data/split_data.py`
- [ ] Verify output folder structure and counts

