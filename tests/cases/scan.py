from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class ScanTestSource(BatchProvider):

    def setup(self):

        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=Roi((20000, 2000, 2000), (2000, 200, 200)),
                voxel_size=(20, 2, 2)))
        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(
                roi=Roi((20100,2010,2010), (1800,180,180)),
                voxel_size=(20, 2, 2)))

    def provide(self, request):

        # print("ScanTestSource: Got request " + str(request))

        batch = Batch()

        # have the pixels encode their position
        for (array_key, spec) in request.array_specs.items():

            roi = spec.roi
            roi_voxel = roi // self.spec[array_key].voxel_size
            # print("ScanTestSource: Adding " + str(array_key))

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(roi_voxel.get_begin()[0], roi_voxel.get_end()[0]),
                    range(roi_voxel.get_begin()[1], roi_voxel.get_end()[1]),
                    range(roi_voxel.get_begin()[2], roi_voxel.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            # print("Roi is: " + str(roi))

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(
                    data,
                    spec)

        return batch

class TestScan(ProviderTest):

    def test_output(self):

        source = ScanTestSource()

        chunk_request = BatchRequest()
        chunk_request.add(ArrayKeys.RAW, (400,30,34))
        chunk_request.add(ArrayKeys.GT_LABELS, (200,10,14))

        pipeline = ScanTestSource() + Scan(chunk_request, num_workers=10)

        with build(pipeline):

            raw_spec = pipeline.spec[ArrayKeys.RAW]
            labels_spec = pipeline.spec[ArrayKeys.GT_LABELS]

            full_request = BatchRequest({
                    ArrayKeys.RAW: raw_spec,
                    ArrayKeys.GT_LABELS: labels_spec
                }
            )

            batch = pipeline.request_batch(full_request)
            voxel_size = pipeline.spec[ArrayKeys.RAW].voxel_size

        # assert that pixels encode their position
        for (array_key, array) in batch.arrays.items():

            # the z,y,x coordinates of the ROI
            roi = array.spec.roi
            meshgrids = np.meshgrid(
                    range(roi.get_begin()[0]//voxel_size[0], roi.get_end()[0]//voxel_size[0]),
                    range(roi.get_begin()[1]//voxel_size[1], roi.get_end()[1]//voxel_size[1]),
                    range(roi.get_begin()[2]//voxel_size[2], roi.get_end()[2]//voxel_size[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            self.assertTrue((array.data == data).all())

        assert(batch.arrays[ArrayKeys.RAW].spec.roi.get_offset() == (20000, 2000, 2000))

        # test scanning with empty request

        pipeline = ScanTestSource() + Scan(chunk_request, num_workers=1)
        with build(pipeline):
            batch = pipeline.request_batch(BatchRequest())
