# encoding: utf-8
"""
@author: hello-lijl
@time: 2023/5/20 19:48
@desc: 
"""
import os
import mitsuba as mi
import numpy as np
import open3d as o3d
import h5py
import time
import matplotlib.pyplot as plt

# replaced by command line arguments
# PATH_TO_NPY = 'pcl_ex.npy' # the tensor to load
# note that sampler is changed to 'independent' and the ldrfilm is changed to hdrfilm
xml_head = \
    """
    <scene version="0.5.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="1920"/>
                <integer name="height" value="1080"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.005"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>

    """

xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="0.00625"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """

def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result

def load_h5(filename):
    with h5py.File(filename, 'r') as f:
        pc_array = np.asarray(f['data'], dtype=np.float32)
    return pc_array

def ply_2_xml(plyfilepath):
    xml_segments = [xml_head]
    basefilename = os.path.basename(plyfilepath)
    filename, file_extension = os.path.splitext(basefilename)

    if (file_extension == '.npy'):
        pcl = np.load(plyfilepath)
    elif (file_extension == '.ply'):
        pcd = o3d.io.read_point_cloud(plyfilepath)
        pcl = np.asarray(pcd.points)
    elif (file_extension == '.h5'):
        pcl = load_h5(plyfilepath)
        # pcl = np.asarray(pcd.points)
    elif (file_extension == '.exr' or file_extension == '.xml' or file_extension == '.png'):
        return
    else:
        raise Exception('unsupported file format.')

    pcl = standardize_bbox(pcl, len(pcl))
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    for i in range(pcl.shape[0]):
        color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
        xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    with open(f'data_xml/{filename}.xml', 'w') as f:
        f.write(xml_content)
    f.close()

def xml_2_exr(xmlfilepath):
    # Set the variant of the renderer
    print(mi.variants())
    mi.set_variant('scalar_rgb')
    # Load a scene
    scene = mi.load_file(xmlfilepath)
    # Render the scene
    image = mi.render(scene)

    plt.axis("off")
    plt.imshow(image ** (1.0 / 2.2))  # approximate sRGB tonemapping
    # Write the rendered image to an EXR file
    
    basefilename = os.path.basename(xmlfilepath)
    filename, file_extension = os.path.splitext(basefilename)
    mi.util.write_bitmap(f"data_png/{filename}.png", image)
    mi.util.write_bitmap(f"data_exr/{filename}.exr", image)

if __name__ == '__main__':
    # ply 2 xml 32
    # plyfilepath = "data_ply/output_pc_036.ply"
    # ply_2_xml(plyfilepath)
    
    # # xml 2 png,exr
    # start = time.time()
    # basefilename = os.path.basename(plyfilepath)
    # filename, file_extension = os.path.splitext(basefilename)
    # xml_2_exr(f"data_xml/{filename}.xml")

    # print(f"The time is {time.time()-start:.6f}")


    # dir
    files = os.listdir("data_ply")
    for item in files:
        filename, file_extension = os.path.splitext(item)
        if file_extension == ".ply":
            print(f"data_ply/{filename}.ply")
            ply_2_xml(f"data_ply/{filename}.ply")
            # xml 2 png,exr
            start = time.time()
            xml_2_exr(f"data_xml/{filename}.xml")

            print(f"The time is {time.time()-start:.6f}")