# comptic: Computational Optical Imaging support library
***Work in progress - API may change without warning***

## Requirements
numpy, scipy, [llops](http://www.github.com/zfphil/llops)

## Installation
```shell
git clone https://www.github.com/zfphil/comptic
cd illuminate_controller
python setup.py build
python setup.py install
```

## Submodule List
- ***comptic.camera***: Camera-related tools for demosaicing, etc.
- ***comptic.containers***: Data structures for datasets and metadata
- ***comptic.dataset***: [Depreciated]
- ***comptic.error***: Error metric functions
- ***comptic.imaging***: Imaging related functions (such as pupil and OTF generation)
- ***comptic.ledarray***: Tools for generating LED array positions
- ***comptic.metadata***: [Depreciated]
- ***comptic.noise***: Tools for adding and measuring noise in images
- ***comptic.prop***: Tools for optical propagation
- ***comptic.ransac***: RANSAC implementations
- ***comptic.registration***: Tools for image registration
- ***comptic.simulation***: Tools for simulating objects
- ***comptic.transformation***: Tools for applying transformations to Images
- ***comptic.wotf***: Tools for generating weak-object transfer functions

## License
BSD 3-clause
