import segmenter as seg

seg.segment_images(['data/T35VME_20200531.tif',
                    'data/T35VME_20200628.tif',
                    'data/T35VME_20200718.tif'],
                   'output/T35VME_seg3.tif')
