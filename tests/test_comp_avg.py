import pytest
import subprocess
import hashlib
import os
import shutil
from os import path as op
import nipype_inc as ca
import nibabel as nib
import numpy as np


chunk  = op.abspath('sample_data/dummy_1.nii')
chunks = [chunk, op.abspath('sample_data/dummy_3.nii')]

def test_increment():
    delay = 0
    benchmark = False
    cli = True
    im = ca.increment_chunk(chunk, delay, benchmark, cli=True)

    assert op.isfile(im)

    original = nib.load(chunk).get_data()
    incremented = nib.load(im).get_data()

    assert np.array_equal(incremented, original + 1)

def test_increment_wf():
    delay = 0
    benchmark = False
    benchmark_dir = None
    cli = True
    wf_name = 'test_incwf'
    avg = None
    work_dir = 'test_incwf_work'
    inc_chunks = ca.increment_wf([chunks], delay, benchmark, benchmark_dir,
                                  cli, wf_name, avg, work_dir)
    for im in inc_chunks:
        assert op.isfile(im)

def test_compute_avg():
    benchmark = False
    benchmark_dir = None
    avg = ca.compute_avg([chunks], benchmark, benchmark_dir)

    images = [i for c in [chunks] for i in c]

    data = None

    for im in images:
        if data is None:
            data = nib.load(im).get_data().astype(np.float64, copy=False)
        else:
            data += nib.load(im).get_data().astype(np.float64, copy=False)

    data = data / len(images)

    assert op.isfile(avg)
    assert np.array_equal(nib.load(avg).get_data(), data.astype(np.uint16))

def test_compute_avg_wf():
    from nipype import Workflow

    nnodes = 1
    work_dir = 'test_ca_wf_work'
    chunk = chunks
    delay = 0
    benchmark_dir = None
    benchmark = False
    cli = True

    wf = Workflow('test_ca_wf')
    wf.base_dir = work_dir

    inc_1, ca_1 = ca.computed_avg_node('ca_bb',
                                       nnodes, work_dir,
                                       chunk=chunk,
                                       delay=delay,
                                       benchmark_dir=benchmark_dir,
                                       benchmark=benchmark,
                                       cli=cli)
        
    wf.add_nodes([inc_1])

    wf.connect([(inc_1, ca_1, [('inc_chunk', 'chunks')])])
    nodename = 'inc_2_test'
    inc_2, ca_2 = ca.computed_avg_node(nodename, nnodes, work_dir, delay=delay,
                                 benchmark_dir=benchmark_dir,
                                 benchmark=benchmark, cli=cli)

    wf.connect([(ca_1, inc_2, [('avg_chunk', 'avg')])])
    wf.connect([(inc_1, inc_2, [('inc_chunk', 'chunk')])])
    wf_out = wf.run('SLURM',
                    plugin_args={
                      'template': 'benchmark_scripts/nipype_kmeans_template.sh'
                    })

    node_names = [i.name for i in wf_out.nodes()]
    result_dict = dict(zip(node_names, wf_out.nodes()))
    saved_chunks = (result_dict['ca1_{0}'.format(nodename)]
                                 .result
                                 .outputs
                                 .inc_chunk)
    avg_file = (result_dict['ca2_{0}'.format('ca_bb')]
                                 .result
                                 .outputs
                                 .avg_chunk)
    inc1_chunks = (result_dict['ca1_{0}'.format('ca_bb')]
                                 .result
                                 .outputs
                                 .inc_chunk)

    results = [i for c in saved_chunks for i in c]
    inc1 = [i for c in inc1_chunks for i in c]

    im_1 = nib.load(chunks[0])
    im_3 = nib.load(chunks[1])
    
    assert np.array_equal(im_3.get_data(), nib.load(chunks[1]).get_data())

    im_1_inc = (im_1.get_data() + 1)
    im_3_inc = (im_3.get_data() + 1)
    nib.save(nib.Nifti1Image(im_1_inc, im_1.affine, im_1.header),
             'test-inc1_1.nii')
    nib.save(nib.Nifti1Image(im_3_inc, im_3.affine, im_3.header),
             'test-inc3_1.nii')

    for i in inc1:
        if 'dummy_1' in i:
            assert np.array_equal(nib.load(i).get_data(),
                                  nib.load('test-inc1_1.nii').get_data())
        else:
            assert np.array_equal(nib.load(i).get_data(),
                                  nib.load('test-inc3_1.nii').get_data())

    avg = None

    for i in [im_1_inc, im_3_inc]:
        if avg is None:
            avg = i.astype(np.float64, casting='safe')
        else:
            avg += i.astype(np.float64, casting='safe')

    avg /= len([im_1_inc, im_3_inc])

    nib.save(nib.Nifti1Image(avg.astype(np.uint16), np.eye(4)), 't_avg.nii')

    assert np.array_equal(nib.load(avg_file).get_fdata(),
                          nib.load('t_avg.nii').get_fdata()) 

    im_1_inc_2 = nib.load('test-inc1_1.nii').get_data() + 1
    im_3_inc_2 = nib.load('test-inc3_1.nii').get_data() + 1

    avg = nib.load('t_avg.nii').get_data()
    im_1_ca = (im_1_inc_2 + avg)
    im_3_ca = (im_3_inc_2 + avg)

    nib.save(nib.Nifti1Image(im_1_ca, im_1.affine, im_1.header), 
            'test-inc1_2.nii')
    nib.save(nib.Nifti1Image(im_3_ca, im_3.affine, im_3.header), 
            'test-inc3_2.nii')


    for i in results:
        assert op.isfile(i)
        ca_res = nib.load(i)
        ca_res = ca_res.get_data().astype(np.uint16)
        if 'inc-dummy_1.nii' in i:
            im = nib.load('test-inc1_2.nii')
            exp = im.get_data().astype(np.uint16)
            assert np.array_equal(ca_res, exp)
        else:
            im = nib.load('test-inc3_2.nii')
            exp = im.get_data().astype(np.uint16)
            assert np.array_equal(ca_res, exp)


    p = subprocess.Popen(['python', 'pipelines/spark_inc.py', 'sample_data',
                          'spca_out', '2', '--benchmark', '--computed_avg'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE)
    (out, err) = p.communicate()

    h_prog_1 = hashlib.md5(open('spca_out/inc2-dummy_1.nii', 'rb').read()) \
                       .hexdigest()
    h_exp_1 = hashlib.md5(open('test-inc1_2.nii', 'rb')
                           .read()) \
                      .hexdigest()

    h_prog_2 = hashlib.md5(open('spca_out/inc2-dummy_3.nii', 'rb').read()) \
                       .hexdigest()
    h_exp_2 = hashlib.md5(open('test-inc3_2.nii', 'rb')
                           .read()) \
                      .hexdigest()

    assert h_prog_1 == h_exp_1
    assert h_prog_2 == h_exp_2
