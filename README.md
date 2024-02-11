<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/thomasnonis/h4vdm">
    <img src="images/logo.webp" alt="Logo" width="150" height="150">
  </a>

<h3 align="center">Implementation of H4VDM</h3>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

![project's structure](/images/structure.png)

This project's goal is to implement the neural network proposed in [H4VDM by Z. Xiang et al.](https://arxiv.org/abs/2210.11549) as part of the Multimedia Data Security course held by prof. Giulia Boato at the University of Trento.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

The following resources are required for the code to work:
* [Python](https://www.python.org/downloads/)
* [Protobuf](https://github.com/protocolbuffers/protobuf#protobuf-compiler-installation) (only if you need to rebuild the protobuffers)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/thomasnonis/h4vdm.git
   ```
2. Install the dependencies
    ```sh
    pip install -r requirements.txt
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Structure
The codebase is divided in 3 main sections:
- H264 extractor
- Packages
- Runners

The H264 section (`h264-extractor`) is a submodule linked to https://gitlab.com/viper-purdue/h4vdm. It consists of a modified version of ffmpeg that packs the H264 parameters from an MP4 video in an easy to use protobuffer. Full credit to the original authors for this section. The only modification has been the regeneration of the python protobuf files with a newer version.

The packages section ('packages') contains the core of the work of this project. It contains all of the classes and network code.

The runners are simply the `.ipynb` files that are actually run to execute the code.


### Dataset generation

The codebase is written to allow the whole dataset to be lazily generated. To speed up the training phase, it is possible to generate it in advance.

If lazy generation is desired, simply run the code with `build_on_init` and `download_on_init` set to `False`, otherwise set them as desired.
Before doing anything, make sure to set your desired dataset generation directory in `packages/constands.py` with the `DATASET_ROOT` folder. Make sure to have at least 200GB available.

When generating the dataset, the structure will be saved in a `.json` file to keep track of the files across multiple runs. To ignore it and forcibly regenerate the dataset or to include additional resources, `ignore_local_dataset` can be set to `True`.

To ease the process, it may be worth looping through the whole dataset to generate it before doing any training work.

### Training

See `h4vdm.ipynb` for a complete example.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] H264 parameter extraction
- [x] H264 dataset management
- [x] Lazy generation
- [x] Minimize RAM usage
- [ ] Training and performance evaluation
- [ ] Randomize crop centering
- [ ] GOP extraction from random location 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GNU GPLv3 license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Thomas Nonis - [thomas.nonis@studenti.unitn.it](mailto:thomas.nonis@studenti.unitn.it)

<p align="right">(<a href="#readme-top">back to top</a>)</p>