# Yolo-CLIP-annoy demo app

### Start

```shell
streamlit run streamlit_demo.py --server.port <PORT> --browser.serverAddress <ADRESS> -- \
  --clip "path/to/clip/model/ViT-B-32.pt" \
  --annoy "path/to/annoy/index/valid_100.ann" \
  --images "path/to/images/dataframe/valid.done.pkl.zip"
```
