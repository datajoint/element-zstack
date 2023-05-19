import datajoint as dj
import numpy as np
import hashlib
import uuid

from element_interface.utils import dict_to_uuid

from . import stack


schema = dj.Schema()


def activate(
    schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True
):
    """Activate this schema.
    Args:
        schema_name (str): Schema name on the database server to activate the
            `stack_matching` schema
        create_schema (bool): When True (default), create schema in the database if it
            does not yet exist.
        create_tables (bool): When True (default), create tables in the database if they
            do not yet exist.
    """
    assert stack.schema.is_activated(), 'The "stack" schema must be activated'
    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=stack.__dict__,
    )


@schema
class StackMatchTask(dj.Manual):
    definition = """  # 
    stack_match_task: uuid
    """

    class Stack(dj.Part):
        definition = """
        -> master
        -> Stack.Segmentation
        """

    @classmethod
    def insert1(cls, *args, **kwargs):
        raise NotImplemented('Use .insert_stack_pair() to insert a StackMatchTask')

    def insert_tack_pair(cls, stack_keys):
        """
        Args:
            stack_keys: a restriction identifying two cell-segmented stacks, e.g. a list of two dicts.
        """
        # validate the stack keys
        stack_keys = (stack.Segmentation & stack_keys).fetch1("KEY")
        assert len(stack_keys) == 2, "Stack match task only supports matching two cell-segmented stacks"

        # create the unique ID for the stack pair
        hashed = hashlib.md5()
        for k in sorted(dict_to_uuid(k) for k in stack_keys):
            hashed.update(k.bytes)
        key = {'stack_match_task': uuid.UUID(hex=hashed.hexdigest())}

        # check if it has already been inserted
        with cls.connection.transaction:
            super().insert1(cls(), key)
            cls.Volume.insert(({**key, **k} for k in vol_seg_keys))


@schema
class StackMatch(dj.Computed):
    definition = """
    -> StackMatchTask
    ---
    execution_time: datetime
    execution_duration: float  # (hours)
    """

    class Transformation(dj.Part):
        definition = """  # transformation matrix
        -> master
        -> VolumeMatchTask.Volume
        ---
        transformation_matrix: longblob  # the transformation matrix to transform to the common space
        """

    class CommonMask(dj.Part):
        definition = """
        common_mask: smallint
        """

    class StackMask(dj.Part):
        definition = """
        -> master.CommonMask
        -> StackMatchTask.Volume
        ---
        -> stack.Segmentation.Mask
        confidence: float
        """

    def make(self, key):
        import point_cloud_registration as pcr

        stack_keys = (stack.Segmentation & (VolumeMatchTask.Volume & key)).fetch('KEY')

        vol1_points, vol2_points = zip(*(stack.Segmentation.Mask & vol_keys).fetch(
            'mask_center_x', 'mask_center_y', 'mask_center_z'))

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
