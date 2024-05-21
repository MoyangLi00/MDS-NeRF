# MDS-NeRF: Neural Radiance Fields with Marigold Depth Supervision
This is the official implementation for our 3DV project.

<div align='center'>
<img src="pipeline.png" height="230px">
</div>

# Installation
LERF follows the integration guidelines described [here](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html) for custom methods within Nerfstudio. 
### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment
### 1. Clone this repo
`https://github.com/MoyangLi00/nope-nerfacto.git`
### 2. Install this repo as a python package
Navigate to this folder and run `python -m pip install -e .`

### 3. Run `ns-install-cli`

### Checking the install
Run `ns-train -h`: you should see a list of "subcommands" with nope-nerfacto.

# Using MDS-NeRF
Now that nope-nerfacto is installed you can play with it! 

- Launch training with `ns-train nope-nerfacto --data <data_folder>`. This specifies a data folder to use. For more details, see [Nerfstudio documentation](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html). 
- Connect to the viewer by forwarding the viewer port (we use VSCode to do this), and click the link to `viewer.nerf.studio` provided in the output of the train script

## Resolution
The Nerfstudio viewer dynamically changes resolution to achieve a desired training throughput.

**To increase resolution, pause training**. Rendering at high resolution (512 or above) can take a second or two, so we recommend rendering at 256px



## Acknowledgment

- Kudos to the [Nerfstudio](https://github.com/nerfstudio-project/) contributors for their amazing work:

```bibtex
@inproceedings{nerfstudio,
	title        = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
	author       = {
		Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent
		and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
		Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,
		Angjoo
	},
	year         = 2023,
	booktitle    = {ACM SIGGRAPH 2023 Conference Proceedings},
	series       = {SIGGRAPH '23}
}
```