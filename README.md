# SketchHealer
# SketchHealer

-   This is a offical repo for **SketchHealer:** **A** **Graph-to-Sequence** **Network** **for** **Recreating** **Partial** **Human** **Sketches**

-   To run this code, you need to install 
    -   pytorch
    -   torchvision
    -   opencv
    -   pillow
    -   and some common dependency libraries.

## Training

-   modify your data location and categories in hyper_params.py

-   ```sh
    python -u sketch_gcn.py
    ```

-   then, model starts training.

## Inference

-   ```sh
    python -u inference.py
    ```

### More Details

More detailed code instructions will be added later. In the meantime, if you have any questions, please contact us.

### Cite

If you have some inspirations for your work, we would appreciate your quoting our paper.
```
@article{su2020sketchhealer,
  title={SketchHealer: A Graph-to-Sequence Network for Recreating Partial Human Sketches},
  author={Su, Guoyao and Qi, Yonggang and Pang, Kaiyue and Yang, Jie and Song, Yi-Zhe},
  journal={BMVC},
  year={2020}
}
```