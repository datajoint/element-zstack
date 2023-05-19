import datajoint as dj
import numpy as np
import hashlib
import uuid

from element_interface.utils import dict_to_uuid

from . import volume


schema = dj.Schema()


def activate(
    schema_name: str, *, create_schema: bool = True, create_tables: bool = True
):
    """Activate this schema.
    Args:
        schema_name (str): Schema name on the database server to activate the
            `volume_matching` schema
        create_schema (bool): When True (default), create schema in the database if it
            does not yet exist.
        create_tables (bool): When True (default), create tables in the database if they
            do not yet exist.
    """
    assert volume.schema.is_activated(), 'The "volume" schema must be activated'
    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=volume.__dict__,
    )


@schema
class VolumeMatchTask(dj.Manual):
    """Defines the volumes and segmentations that will be registered in the downstream `VolumeMatch` table.

    Attributes:
        volume_match_task (uuid): UUID hash for the volume matching task
    """

    definition = """  # 
    volume_match_task: uuid
    """

    class Volume(dj.Part):
        """Defines the volume segmentations that will be registered in the downstream `VolumeMatch` table.

        Attributes:
            VolumeMatchTask (foreign key): Primary key from `VolumeMatchTask`.
            volume.Segmentation (foreign key): Primary key from `volume.Segmentation`.
        """

        definition = """
        -> master
        -> volume.Segmentation
        """

    @classmethod
    def insert1(cls, *args, **kwargs):
        raise NotImplemented(
            'Use .insert_stack_pair() to insert a StackMatchTask')

    def insert_stack_pair(self, stack_keys):
        """Insert an entry into the `VolumeMatchTask` and `VolumeMatchTask.Volume` tables.

        Args:
            stack_keys: a restriction specifying a pair of segmented volumes
        """

        # validate the stack keys
        stack_keys = (volume.Segmentation & stack_keys).fetch1("KEY")
        assert len(
            stack_keys) == 2, "Stack match task only supports matching two cell-segmented stacks"

        # create a volume pair id
        hashed = hashlib.md5()
        for k in sorted(dict_to_uuid(k) for k in stack_keys):
            hashed.update(k.bytes)
        key = {'stack_match_task': uuid.UUID(hex=hashed.hexdigest())}

        # insert the volume pair
        with self.connection.transaction:
            self.insert1(key)
            self.Volume.insert({**key, **k} for k in stack_keys)


@schema
class VolumeMatch(dj.Computed):
    """Execute the volume matching algorithm and store the results.

    Attributes:
        VolumeMatchTask (foreign key): Primary key from `VolumeMatchTask`.
        execution_time (datetime): Execution time of the volume matching task.
        execution_duration (float): Duration of the volume matching task.
    """

    definition = """
    -> VolumeMatchTask
    ---
    execution_time: datetime
    execution_duration: float  # (hr)
    """

    class Transformation(dj.Part):
        """Store the transformation matrix to the common space.

        Attributes:
            VolumeMatch (foreign key): Primary key from `VolumeMatch`.
            VolumeMatchTask.Volume (foreign key): Primary key from `VolumeMatchTask.Volume`.
            transformation_matrix (longblob): the transformation matrix to
            transform to the common space.
        """

        definition = """  # transformation matrix
        -> master
        -> VolumeMatchTask.Volume
        ---
        transformation_matrix: longblob  # the transformation matrix to transform to the common space
        """

    class CommonMask(dj.Part):
        """Store common mask identifier.

        Attributes:
            common_mask (smallint): Integer value for the common mask identifier.
        """

        definition = """
        common_mask: smallint
        """

    class VolumeMask(dj.Part):
        """For the masks in the common space, store the associated mask in the segmented volumes, and the confidence of the volume registration and cell matching.

        Attributes:
            VolumeMatch.CommonMask (foreign key): Primary key from
            `VolumeMatch.CommonMask`.
            VolumeMatchTask.Volume (foreign key): Primary key from
            `VolumeMatchTask.Volume`.
            volume.Segmentation.Mask (foreign key): Primary key from
            `volume.Segmentation.Mask`.
            confidence (float): confidence in the volume registration and cell matching between the segmented volumes.
        """

        definition = """
        -> master.CommonMask
        -> VolumeMatchTask.Volume
        ---
        -> volume.Segmentation.Mask
        confidence: float
        """

    def make(self, key):
        import point_cloud_registration as pcr
        from scipy.stats import gaussian_kde

        vol_keys = (volume.Segmentation & (
            VolumeMatchTask.Volume & key)).fetch("KEY")

        vol1_points, vol2_points = zip(
            *(volume.Segmentation.Mask & vol_keys).fetch(
                "mask_center_x", "mask_center_y", "mask_center_z"
            )
        )

        vol1_points = np.hstack([*vol1_points])
        vol2_points = np.hstack([*vol2_points])

        tetras1 = pcr.make_normal_tetras(vol1_points)
        tetras2 = pcr.make_normal_tetras(vol2_points)

        pcr.compute_canonical_features(tetras1)
        pcr.remove_common_tetras(tetras1)

        pcr.compute_canonical_features(tetras2)
        pcr.remove_common_tetras(tetras2)

        distances, matches = pcr.match_features(tetras1, tetras2)

        # add complete set of steps once point-cloud-registration algorithm is complete
