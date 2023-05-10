# Data Pipeline

Element ZStack is composed of two main schemas, `volume` and `volume_matching`. Data 
export to BossDB is handled with a `bossdb` schema and upload utilities.

- `volume` module - performs segmentation of volumetric microscopic images with 
`cellpose`.

- `volume_matching` module - performs volume registration to a common space and matches 
cells across imaging sessions.

- `bossdb` module - uploads data to BossDB, creates a Neuroglancer visualization, and 
stores the relevant URLs.

Each node in the following diagram represents the analysis code in the pipeline and the
corresponding table in the database.  Within the workflow, Element ZStack
connects to upstream Elements including Lab, Animal, Session, and Calcium Imaging. For 
more detailed documentation on each table, see the API docs for the respective schemas.

![pipeline](https://raw.githubusercontent.com/datajoint/element-zstack/images/pipeline.svg)

### `reference` schema ([API docs](https://datajoint.com/docs/elements/element-zstack/latest/api/workflow_zstack/reference))

| Table | Description |
| --- | --- |
| Device | Lab equipment metadata |

### `subject` schema ([API docs](https://datajoint.com/docs/elements/element-animal/latest/api/element_animal/subject))

- Although not required, most choose to connect the `Session` table to a `Subject` table.

| Table | Description |
| --- | --- |
| Subject | Basic information of the research subject |

### `session` schema ([API docs](https://datajoint.com/docs/elements/element-session/latest/api/element_session/session_with_datetime))

| Table | Description |
| --- | --- |
| Session | Unique experimental session identifier |

### `scan` schema ([API docs](https://datajoint.com/docs/elements/element-calcium-imaging/latest/api/element_calcium_imaging/scan))

| Table | Description |
| --- | --- |
| Scan | A set of imaging scans performed in a single session |

### `volume` schema ([API docs](https://datajoint.com/docs/elements/element-zstack/latest/api/element_zstack/volume))

| Table | Description |
| --- | --- |
| Volume | Details about the volumetric microscopic images |
| VoxelSize | Voxel size information about a volume in millimeters |
| SegmentationParamSet | Parameter set used for segmentation of the volumetric microscopic images |
| SegmentationTask | Defines the method and parameter set which will be used to segment a volume in the downstream `Segmentation` table |
| Segmentation | Performs segmentation on the volume (and with the method and parameter set) defined in the `SegmentationTask` table |
| Segmentation.Mask | Details of the masks identified from the segmentation |

### `volume_matching` schema ([API docs](https://datajoint.com/docs/elements/element-zstack/latest/api/element_zstack/volume_matching))

| Table | Description |
| --- | --- |
| VolumeMatchTask | Defines the volumes and segmentations that will be registered in the downstream `VolumeMatch` table |
| VolumeMatchTask.Volume | Defines the volume segmentations that will be registered in the downstream `VolumeMatch` table |
| VolumeMatch | Execute the volume matching algorithm and store the results |
| VolumeMatch.Transformation | Store the transformation matrix to the common space |
| VolumeMatch.CommonMask | Store common mask identifier |
| VolumeMatch.VolumeMask | For the masks in the common space, store the associated mask in the segmented volumes, and the confidence of the volume registration and cell matching |

### `bossdb` schema ([API docs](https://datajoint.com/docs/elements/element-zstack/latest/api/element_zstack/bossdb))

| Table | Description |
| --- | --- |
| VolumeUploadTask | Define the image and segmentation data to upload to BossDB |
| VolumeUpload | Upload image and segmentation data to BossDB, and store the BossDB and Neuroglancer URLs |