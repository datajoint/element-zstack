import importlib
import inspect
import pathlib
import datajoint as dj
import numpy as np

from tqdm import tqdm
from tifffile import TiffFile
from typing import Union

from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory


logger = dj.logger

schema = dj.Schema()
_linking_module = None


def activate(
    schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema

    Args:
        schema_name (str): schema name on the database server to activate the `zstack` element
        create_schema (bool): when True (default), create schema in the database
        if it does not yet exist.
        create_tables (bool): when True (default), create schema tables in the database if they do not yet exist.
        linking_module (str): A string containing the module name or module
        containing the required dependencies to activate the schema.

    Tables:
        Scan: A parent table to Volume
    Functions:
        get_volume_root_data_dir: Returns absolute path for root data
        director(y/ies) with all volumetric data, as a list of string(s).
        get_volume_tif_file: When given a scan key (dict), returns the full path
        to the TIF file of the volumetric data associated with a given scan.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'linking_module' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# -------------------------- Functions required by the Element -------------------------


def get_volume_root_data_dir() -> list:
    """Fetches absolute data path to volume data directories.

    The absolute path here is used as a reference for all downstream relative paths used in DataJoint.

    Returns:
        A list of the absolute path(s) to volume data directories.
    """
    root_directories = _linking_module.get_volume_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    return root_directories


def get_processed_root_data_dir() -> Union[str, pathlib.Path]:
    """Retrieve the root directory for all processed data.

    All data paths and directories in DataJoint Elements are recommended to be stored as
    relative paths (posix format), with respect to some user-configured "root"
    directory, which varies from machine to machine (e.g. different mounted drive
    locations).

    Returns:
        dir (str| pathlib.Path): Absolute path of the processed imaging root data
            directory.
    """

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        return _linking_module.get_processed_root_data_dir()
    else:
        return get_volume_root_data_dir()[0]


def get_volume_tif_file(scan_key: dict) -> Union[str, pathlib.Path]:
    """Retrieve the full path to the TIF file of the volumetric data associated with a given scan.
    Args:
        scan_key: Primary key of a Scan entry.
    Returns:
        Full path to the TIF file of the volumetric data (Path or str).
    """
    return _linking_module.get_volume_tif_file(scan_key)


# --------------------------------------- Schema ---------------------------------------


@schema
class Volume(dj.Imported):
    """Details about the volumetric microscopic imaging scans.

    Attributes:
        Scan (foreign key): Primary key from `imaging.Scan`.
        px_width (int): total number of voxels in the x dimension.
        px_height (int): total number of voxels in the y dimension.
        px_depth (int): total number of voxels in the z dimension.
        depth_mean_brightness (longblob): optional, mean brightness of each slice across
        the depth (z) dimension of the stack.
        volume_file_path (str): Relative path of the volumetric data with shape (z, y, x)
    """

    definition = """
    -> Scan
    ---
    px_width: int # total number of voxels in x dimension
    px_height: int # total number of voxels in y dimension
    px_depth: int # total number of voxels in z dimension
    depth_mean_brightness=null: longblob  # mean brightness of each slice across the depth (z) dimension of the stack
    volume_file_path: varchar(255)  # Relative path of the volumetric data with shape (z, y, x)
    """

    def make(self, key):
        """Populate the Volume table with volumetric microscopic imaging data."""
        volume_file_path = get_volume_tif_file(key)
        volume_data = TiffFile(volume_file_path).asarray()

        root_dir = find_root_directory(get_volume_root_data_dir(), volume_file_path)
        volume_relative_path = (
            pathlib.Path(volume_file_path).relative_to(root_dir).as_posix()
        )

        self.insert1(
            dict(
                **key,
                volume_file_path=volume_relative_path,
                px_width=volume_data.shape[2],
                px_height=volume_data.shape[1],
                px_depth=volume_data.shape[0],
                depth_mean_brightness=volume_data.mean(axis=(1, 2)),
            )
        )


@schema
class VoxelSize(dj.Manual):
    """Voxel size information about a volume in millimeters.

    Attributes:
        Volume (foreign key): Primary key from `Volume`.
        width (float): Voxel size in mm in the x dimension.
        height (float): Voxel size in mm in the y dimension.
        depth (float): Voxel size in mm in the z dimension.
    """

    definition = """
    -> Volume
    ---
    width: float # voxel size in mm in the x dimension
    height: float # voxel size in mm in the y dimension
    depth: float # voxel size in mm in the z dimension
    """


@schema
class SegmentationMethod(dj.Lookup):
    """Segmentation methods used for processing volume data.

    Attributes:
        segmentation_method (str): Name of the segmentation method (e.g. cellpose).
        segmentation_method_desc (str): Optional. Description of the segmentation method.
    """

    definition = """
    segmentation_method: varchar(32)
    ---
    segmentation_method_desc: varchar(1000)
    """

    contents = [
        ("cellpose", "cellpose analysis suite"),
    ]


@schema
class SegmentationParamSet(dj.Lookup):
    """Parameter set used for segmentation of the volumetric microscopic imaging
    scan.

    Attributes:
        paramset_idx (int): Unique parameter set identifier.
        SegmentationMethod (foreign key): Primary key from `SegmentationMethod`.
        paramset_desc (str): Parameter set description.
        params (longblob): Parameter set. Dictionary of all applicable
        parameters for the segmentation method.
        paramset_hash (uuid): A universally unique identifier for the parameter set.
    """

    definition = """
    paramset_idx: int # Unique parameter set ID.
    ---
    -> SegmentationMethod
    paramset_desc: varchar(1000)
    params: longblob # dictionary of all applicable parameters for the segmentation method.
    paramset_hash: uuid # A universally unique identifier for the parameter set.
    """

    @classmethod
    def insert_new_params(
        cls,
        segmentation_method: str,
        params: dict,
        paramset_desc: str,
        paramset_idx: int,
    ):
        """Inserts new parameters into the SegmentationParamSet table.

        This function automates the parameter set hashing and avoids insertion of an existing parameter set.

        Args:
            segmentation_method (str): name of the segmentation method (e.g. cellpose)
            paramset_idx (int): Unique parameter set ID.
            paramset_desc (str): Description of the parameter set.
            params (dict): segmentation parameters
        """

        param_dict = {
            "segmentation_method": segmentation_method,
            "paramset_desc": paramset_desc,
            "params": params,
            "paramset_idx": paramset_idx,
            "paramset_hash": dict_to_uuid(
                {**params, "segmentation_method": segmentation_method}
            ),
        }
        param_query = cls & {"paramset_hash": param_dict["paramset_hash"]}

        if param_query:  # If the specified param-set already exists
            existing_paramset_idx = param_query.fetch1("paramset_idx")
            if (
                existing_paramset_idx == paramset_idx
            ):  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    f"The specified param-set already exists"
                    f" - with paramset_idx: {existing_paramset_idx}"
                )
        else:
            if {"paramset_idx": paramset_idx} in cls.proj():
                raise dj.DataJointError(
                    f"The specified paramset_idx {paramset_idx} already exists,"
                    f" please pick a different one."
                )
            cls.insert1(param_dict)


@schema
class SegmentationTask(dj.Manual):
    """Defines the method and parameter set which will be used to segment a volume in the downstream `Segmentation` table.  This table currently supports triggering segmentation with `cellpose`.

    Attributes:
        Volume (foreign key): Primary key from `Volume`.
        SegmentationParamSet (foreign key): Primary key from
        `SegmentationParamSet`.
        segmentation_output_dir (str): Optional. Output directory of the
        segmented results relative to the root data directory.
        task_mode (enum): `Trigger` computes segmentation or `load` imports existing results.
    """

    definition = """
    -> Volume
    -> SegmentationParamSet
    ---
    segmentation_output_dir: varchar(255)  #  Output directory of the segmented results relative to root data directory
    task_mode='load': enum('load', 'trigger') # 'load' imports existing results, 'trigger' computes segmentation
    """

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False):
        """Infer an output directory for an entry in SegmentationTask table.

        Args:
            key (dict): Primary key from the SegmentationTask table.
            relative (bool): If True, segmentation_output_dir is returned relative to
                volume_root_dir. Default False.
            mkdir (bool): If True, create the segmentation_output_dir directory.
                Default True.

        Returns:
            dir (str): A default output directory for the processed results (segmentation_output_dir
                in SegmentationTask) based on the following convention:
                processed_dir / scan_dir / {segmentation_method}_{paramset_idx}
                e.g.: sub4/sess1/scan0/suite2p_0
        """
        scan_dir = find_full_path(
            get_volume_root_data_dir(),
            get_volume_tif_file(key),
        ).parent
        root_dir = find_root_directory(get_volume_root_data_dir(), scan_dir)

        method = (
            (SegmentationParamSet & key).fetch1("segmentation_method").replace(".", "-")
        )

        processed_dir = pathlib.Path(get_processed_root_data_dir())
        output_dir = (
            processed_dir
            / scan_dir.relative_to(root_dir)
            / f'{method}_{key["paramset_idx"]}'
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def generate(cls, scan_key, paramset_idx=0):
        """Generate a SegmentationTask for a Scan using an parameter SegmentationParamSet

        Generate an entry in the SegmentationTask table for a particular scan using an
        existing parameter set from the SegmentationParamSet table.

        Args:
            scan_key (dict): Primary key from Scan table.
            paramset_idx (int): Unique parameter set ID.
        """
        key = {**scan_key, "paramset_idx": paramset_idx}

        processed_dir = get_processed_root_data_dir()
        output_dir = cls.infer_output_dir(key, relative=False, mkdir=True)

        method = (SegmentationParamSet & {"paramset_idx": paramset_idx}).fetch1(
            "segmentation_method"
        )

        if not method == "cellpose":
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))
        if any(output_dir.glob("*_seg.npy")):
            task_mode = "load"
        else:
            task_mode = "trigger"

        cls.insert1(
            {
                **key,
                "processing_output_dir": output_dir.relative_to(
                    processed_dir
                ).as_posix(),
                "task_mode": task_mode,
            }
        )

    auto_generate_entries = generate


@schema
class Segmentation(dj.Computed):
    """Performs segmentation on the volume (and with the method and parameter set) defined in the `SegmentationTask` table.

    Attributes:
        SegmentationTask (foreign key): Primary key from `SegmentationTask`.
    """

    definition = """
    -> SegmentationTask
    """

    class Mask(dj.Part):
        """Details of the masks identified from the segmentation.

        Attributes:
            Segmentation (foreign key): Primary key from `Segmentation`.
            mask (int): Unique mask identifier.
            mask_npix (int): Number of pixels in the mask.
            mask_center_x (float): Center x coordinate in pixels.
            mask_center_y (float): Center y coordinate in pixels.
            mask_center_z (float): Center z coordinate in pixels.
            mask_xpix (longblob): X coordinates in pixels.
            mask_ypix (longblob): Y coordinates in pixels.
            mask_zpix (longblob): Z coordinates in pixels.
            mask_weights (longblob): Weights of the mask at the indices above.
        """

        definition = """ # A mask produced by segmentation.
        -> master
        mask            : smallint
        ---
        mask_npix       : int       # number of pixels in ROIs
        mask_center_x   : float     # X component of the 3D mask centroid in pixel units
        mask_center_y   : float     # Y component of the 3D mask centroid in pixel units
        mask_center_z   : float     # Z component of the 3D mask centroid in pixel units
        mask_xpix       : longblob  # x coordinates in pixels units
        mask_ypix       : longblob  # y coordinates in pixels units
        mask_zpix       : longblob  # z coordinates in pixels units
        mask_weights    : longblob  # weights of the mask at the indices above
        """

    def make(self, key):
        """Populate the Segmentation and Segmentation.Mask tables with results of cellpose segmentation."""

        task_mode, output_dir, params = (
            SegmentationTask * SegmentationParamSet & key
        ).fetch1("task_mode", "segmentation_output_dir", "params")

        if not output_dir:
            output_dir = SegmentationTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            SegmentationTask.update1(
                {**key, "segmentation_output_dir": output_dir.as_posix()}
            )

        try:
            output_dir = find_full_path(
                get_processed_root_data_dir(), output_dir
            ).as_posix()
        except FileNotFoundError as e:
            if task_mode == "trigger":
                processed_dir = pathlib.Path(get_processed_root_data_dir())
                output_dir = processed_dir / output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise e

        if task_mode == "trigger":
            from cellpose import io, models as cellpose_models

            volume_relative_path = (Volume & key).fetch1("volume_file_path")
            volume_file_path = find_full_path(
                get_volume_root_data_dir(), volume_relative_path
            ).as_posix()
            volume_data = TiffFile(volume_file_path).asarray()

            model = cellpose_models.CellposeModel(model_type=params["model_type"])
            cellpose_results = model.eval(
                [volume_data],
                diameter=params["diameter"],
                channels=params.get("channels", [[0, 0]]),
                min_size=params["min_size"],
                z_axis=0,
                do_3D=params["do_3d"],
                anisotropy=params["anisotropy"],
                progress=True,
            )
            masks, flows, _ = cellpose_results
            masks = masks[0]

            io.masks_flows_to_seg(
                [volume_data],
                masks,
                flows[0],
                str(
                    str(output_dir)
                    + f"/{key['subject']}_session_{key['session_id']}_scan_{key['scan_id']}"
                ),
                channels=params.get("channels", [[0, 0]]),
                diams=[params["diameter"]],
            )

        else:
            seg_file_location = list(pathlib.Path(output_dir).rglob("*_seg.npy"))
            if not seg_file_location:
                raise FileNotFoundError(
                    f"No cellpose segmentation file found at {output_dir}"
                )
            if len(seg_file_location) > 1:
                raise ValueError(
                    f"Multiple possible cellpose segmentation files found at {output_dir}. Expected 1."
                )

            segmentation_data = np.load(seg_file_location[0], allow_pickle=True).item()
            masks = segmentation_data["masks"]

        self.insert1({**key})
        for mask_id in tqdm(set(masks.flatten()) - {0}):
            mask = np.argwhere(masks == mask_id)
            mask_npix = mask.shape[0]
            mask_center_z, mask_center_y, mask_center_x = (
                mask[:, 0].mean(axis=0),
                mask[:, 1].mean(axis=0),
                mask[:, 2].mean(axis=0),
            )
            mask_zpix, mask_ypix, mask_xpix = mask[:, 0], mask[:, 1], mask[:, 2]
            mask_entry = {
                **key,
                "mask": mask_id,
                "mask_npix": mask_npix,
                "mask_center_x": mask_center_x,
                "mask_center_y": mask_center_y,
                "mask_center_z": mask_center_z,
                "mask_xpix": mask_xpix,
                "mask_ypix": mask_ypix,
                "mask_zpix": mask_zpix,
                "mask_weights": np.ones(mask_npix),
            }
            self.Mask.insert1(mask_entry)
