import numpy as np
import matplotlib.pyplot as plt
import math
n_jobs = 10

def do_downsampling(vertice,precentage):
    total_num = vertice.shape[0]
    sampled_num = int(total_num*precentage)
    sampled_id =np.random.choice(total_num, sampled_num, replace=False)
    sampled_id = np.sort(sampled_id)
    sampled_vertice  = vertice[sampled_id,:]
    return sampled_id,sampled_vertice




def do_linear_pca(vertice,dim=3.):
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=2,degree=dim, n_jobs=None)
    embedded = kpca.fit_transform(vertice)
    print("the shape of the embedded is {}".format(embedded.shape))
    return embedded



def do_visualization_on_3d(vertice,sample_num=1000,bias=2500,factor=-1,c=None):
    from mpl_toolkits import mplot3d
    ax = plt.axes(projection='3d')
    xdata,ydata,zdata=zip(*vertice)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds')
    st_id = min(factor*(sample_num)+bias,factor*(1+bias))
    ed_id = max(factor*(sample_num)+bias,factor*(1+bias))
    if c is None:
        ax.scatter3D(xdata[st_id:ed_id], ydata[st_id:ed_id], zdata[st_id:ed_id], c=xdata[1:sample_num], cmap='Blues')
    else:
        ax.scatter3D(xdata[st_id:ed_id], ydata[st_id:ed_id], zdata[st_id:ed_id], c=c[1:sample_num], cmap='Blues')
    plt.axis('off')
    plt.show()

def do_visualization_embedded_on_3d(embedded,sample_num=1000, bias=2500,factor=-1,c=None):
    vertice = np.zeros([embedded.shape[0], 3])
    vertice[:,0:2] = embedded
    do_visualization_on_3d(vertice, sample_num, bias, factor,c)


def do_visualization_colorized_on_3d(vertice):
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    xdata,ydata,zdata=zip(*vertice)
    ax.scatter3D(xdata, ydata, zdata, c=ydata, cmap=plt.cm.Spectral)
    plt.show()
def do_visualization_on_2d_thickness(embedded,thickness):
    from mpl_toolkits import mplot3d
    ax = plt.axes(projection='3d')
    xdata, ydata = zip(*embedded)
    ax.scatter3D(xdata, ydata, thickness, c=thickness, cmap='Reds')
    plt.show()


def do_visualization_colorized_on_2d(vertice, c=None):
    import matplotlib.pyplot as plt
    x,y = zip(*vertice)
    plt.plot(x,y, 'ro',markersize=1,linewidth=2)
    if c is not None:
        plt.scatter(c[0], c[1], c=c[2])
    plt.axis('equal')

    plt.show()


def __generate_mask(vertice,mask,interx,intery,startx,starty):
    x,y = zip(*vertice)
    for i in range(len(vertice)):
        idx = math.floor((x[i]-startx)/interx)
        idy = math.floor((y[i]-starty)/intery)
        mask[idy,idx]=False
        for j in range(idx-1,idx+2):
            for k in range(idy-1,idy+2):
                mask[k,j]=False
    return mask

def do_grid_interpolation(vertice,thickness,ninter= 100, plot_img=True):
    from scipy.interpolate import griddata
    xmin, xmax, ymin,ymax = min(vertice[:,0]),max(vertice[:,0]),min(vertice[:,1]),max(vertice[:,1])
    rangex = xmax - xmin
    rangey = ymax - ymin
    interx = rangex/ninter
    intery = rangey/ninter
    xi = np.arange(xmin-interx*5, xmax+interx*5, interx)
    yi = np.arange(ymin-intery*5, ymax+intery*5, intery)
    xi, yi = np.meshgrid(xi, yi)
    mask = np.zeros_like(xi)==0
    mask = __generate_mask(vertice,mask,interx,intery,xi.min(),yi.min())
    x, y = zip(*vertice)
    z = thickness
    zi = griddata((x, y), z, (xi, yi), method='linear')
    zi[mask] =100



    if plot_img:
        contour_num = 80
        maxz = max(z)
        plt.contourf(xi, yi, zi, np.arange(0.01,maxz + maxz/contour_num, maxz/contour_num))
        plt.axis('equal')
        plt.xlabel('xi', fontsize=16)
        plt.ylabel('yi', fontsize=16)

        plt.show()

def __map_thickness_to_2D_projection(embedded, thickness, ninter=100, min_thickness=-1, fpth=None):
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt

    # x = embedded[:,0]
    # y = embedded[:,1]
    # x = x / np.pi * 180
    # x = equal_scale(x, y)
    # embedded[:, 0] = x
    xmin, xmax, ymin, ymax = min(embedded[:, 0]), max(embedded[:, 0]), min(embedded[:, 1]), max(embedded[:, 1])
    rangex = xmax - xmin
    rangey = ymax - ymin
    interx = rangex / ninter
    intery = rangey / ninter
    print(interx, intery)
    xi = np.arange(xmin - interx * 5, xmax + interx * 5, interx)
    yi = np.arange(ymin - intery * 5, ymax + intery * 5, intery)
    xi, yi = np.meshgrid(xi, yi)
    mask = np.zeros_like(xi) == 0
    mask = __generate_mask(embedded, mask, interx, intery, xi.min(), yi.min())
    x, y = zip(*embedded)




    if min_thickness>0:
        thickness[thickness<min_thickness]=min_thickness
    z = thickness
    zi = griddata((x, y), z, (xi, yi), method='linear')


    zi[mask] = -1
    print("the sum of nan value of non masked zi is {}".format(np.sum(np.isnan(zi))))

    #zi[np.isnan(zi)]=0

    # zi now has the thickness, we can write this out as a numpy file
    # np.save(file=fpth, arr=zi)
    print(np.sum(np.isnan(zi)))

    contour_num = 80
    maxz = max(z)
    print("the sum of zero in zi is {}".format(np.sum(zi==0)))
    plt.contourf(xi, yi, zi, np.arange(0.00, maxz + maxz / contour_num, maxz / contour_num))
    plt.axis('equal')
    # plt.plot(x, y, 'ro', markersize=1, linewidth=2)
    plt.xlabel('xi', fontsize=16)
    plt.ylabel('yi', fontsize=16)
    plt.colorbar().ax.tick_params(labelsize=10)
    font = {'size': 10}
    plt.axis('off')

    # zi[mask] = 0
    # zi[np.isnan(zi)]=0
    # print(np.sum(zi))
    if fpth is not None:
        plt.savefig(fpth,dpi=300)
        plt.close('all')
    else:
        plt.show()
        plt.clf()

def scale_z(vertice, scale):
    vertice[:,2] = vertice[:,2]-np.min(vertice[:,2])
    vertice[:len(vertice) // 2,2] = vertice[:len(vertice) // 2,2]/scale
    vertice[len(vertice) // 2:,2] = vertice[len(vertice) // 2:,2]*scale
    return vertice


def rotate_z(vertice, angle, is_fc=True):
    from scipy.spatial.transform import Rotation as R
    if is_fc:
        r = R.from_euler('y', angle, degrees=True)
        rot_mat = r.as_dcm()
        vertice = np.matmul(vertice,rot_mat)
    return vertice




def compute_least_squaure_circle(x, y):
    method_2b = "leastsq with jacobian"
    from scipy import optimize

    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x) / Ri  # dR/dxc
        df2b_dc[1] = (yc - y) / Ri  # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

    Ri_2b = calc_R(*center_2b)
    R_2b = Ri_2b.mean()
    return center_2b, R_2b

def equal_scale(input,ref):
    input = (input - np.min(input))/(np.max(input)-np.min(input))
    input = input*(np.max(ref)-np.min(ref))*1.5+np.min(ref)
    return input

def extract_suface_vertice_from_cylinder(circle, n_sample,z_range):
    """
    :param circle: (cx, cy), radius
    :param n_sample:  num of samples
    :param z_range: z axis
    :return:
    """
    center, r = circle
    phi = np.random.rand(n_sample)*np.pi*2
    z = np.random.rand(n_sample)*(z_range[1]-z_range[0])+ z_range[0]
    vertice = np.stack([r*np.cos(phi)+center[0], r*np.sin(phi)+center[1],z],1)
    return vertice

def visualize_surface_and_cylinder(sur_vertice, cyl_vertice):
    from mpl_toolkits import mplot3d

    ax = plt.axes(projection='3d')
    xdata, ydata, zdata = zip(*sur_vertice)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds')
    xdata, ydata, zdata = zip(*cyl_vertice)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Blues')
    plt.axis('off')
    plt.show()




def dist (p0,p1,p2):
    dim = len(p0[0])
    if dim ==2:
        d01 = np.sqrt((p0[:,0]-p1[:,0])**2 + (p0[:,1]-p1[:,1])**2)
        d02 = np.sqrt((p0[:,0]-p2[:,0])**2 + (p0[:,1]-p2[:,1])**2)
        d12 = np.sqrt((p1[:,0]-p2[:,0])**2 + (p1[:,1]-p2[:,1])**2)
    if dim ==3:
        d01 = np.sqrt((p0[:, 0] - p1[:, 0]) ** 2 + (p0[:, 1] - p1[:, 1]) ** 2 +(p0[:, 2] - p1[:, 2]) ** 2)
        d02 = np.sqrt((p0[:, 0] - p2[:, 0]) ** 2 + (p0[:, 1] - p2[:, 1]) ** 2 +(p0[:, 2] - p2[:, 2]) ** 2)
        d12 = np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2 +(p1[:, 2] - p2[:, 2]) ** 2)
    return d01, d02, d12


def triangle_area(p0,p1,p2):
    p0 = p0.astype(np.float64)
    p1 = p1.astype(np.float64)
    p2 = p2.astype(np.float64)
    x, y, z = dist(p0,p1,p2)
    s = (x+y+z)/2
    a = np.sqrt(s*(s-x)*(s-y)*(s-z))
    a = a.astype(np.float32)
    return a


def triangle_area_2d(p0,p1,p2):
    p0 = p0.astype(np.float64)
    p1 = p1.astype(np.float64)
    p2 = p2.astype(np.float64)
    x1, y1, x2, y2, x3, y3 = p0[:,0],p0[:,1],p1[:,0],p1[:,1],p2[:,0],p2[:,1]
    a = abs(0.5 * (((x2-x1)*(y3-y1))-((x3-x1)*(y2-y1))))
    a = a.astype(np.float32)
    return a

def triangle_area_3d(p0,p1,p2):
    p0 = p0.astype(np.float64)
    p1 = p1.astype(np.float64)
    p2 = p2.astype(np.float64)
    x1, y1, z1, x2, y2, z2,  x3, y3, z3 = p0[:,0],p0[:,1],p0[:,2],p1[:,0],p1[:,1],p1[:,2],p2[:,0],p2[:,1],p2[:,2]
    x12 = x1 - x2
    x13 = x1 - x3
    y12 = y1 - y2
    y13 = y1 - y3
    z12 = z1 - z2
    z13 = z1 - z3
    a =  0.5 * np.sqrt((y12*z13-z12*y13)**2 + (z12*x13 - x12*z13)**2 + (x12*y13-y12*x13)**2)
    a = a.astype(np.float32)
    return a


def compute_embeded_weight(vertice,faces,proj,ninter,ratio_max=10):
    """
    :param vetrice: Nx3 np
    :param faces:  Nx3 np
    :param projs: Nx2 np
    :return: N np
    """
    vertice_faces_sum = np.zeros(len(vertice))
    proj_faces_sum = np.zeros(len(proj))
    vertice_p0=vertice[faces[:,0]]
    vertice_p1=vertice[faces[:,1]]
    vertice_p2=vertice[faces[:,2]]
    vertice_area = triangle_area_3d(vertice_p0,vertice_p1,vertice_p2)
    proj_p0 = proj[faces[:, 0]]
    proj_p1 = proj[faces[:, 1]]
    proj_p2 = proj[faces[:, 2]]
    proj_area = triangle_area_2d(proj_p0, proj_p1, proj_p2)
    for i in range(3):
        for id in range(len(faces)):
            vertice_faces_sum[faces[id,i]] += vertice_area[id]
            proj_faces_sum[faces[id,i]] += proj_area[id]
    ratio = vertice_faces_sum/proj_faces_sum
    #ratio[np.isinf(ratio)] = np.nan
    ratio[np.isinf(ratio)]=ratio_max
    ratio[ratio>ratio_max] =ratio_max

    xmin, xmax, ymin, ymax =min(proj[:, 0]), max(proj[:, 0]), min(proj[:, 1]), max(proj[:, 1])
    rangex = xmax - xmin
    rangey = ymax - ymin
    interx = rangex / ninter
    intery = rangey / ninter
    density = (interx*intery)
    surface_area = density*ratio

    return surface_area




def get_projection_from_circle_and_vertice(vertice, circle):
    center, r =  circle
    x, y = vertice[:,0],vertice[:,1]
    radian = np.arctan2(y - center[1], x - center[0])

    embedded = np.zeros([len(vertice), 2])
    embedded[:, 0] = radian
    embedded[:, 1] = vertice[:, 2]

    plot_xy = np.zeros_like(embedded)
    angle = radian / np.pi * 180
    angle = equal_scale(angle, vertice[:, 2])
    plot_xy[:, 0] = angle
    plot_xy[:, 1] = vertice[:, 2]
    return embedded, plot_xy

def rotate_embedded(embedded,angle):
    theta = (angle / 180.) * np.pi
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
    embedded = c = np.dot(embedded, rotMatrix)
    return embedded
def get_combined_TC_embedding(embedded_left,embedded_right,index_left, index_right):
    embedded = np.zeros([len(embedded_left)+len(embedded_right),2])
    embedded[index_left] = embedded_left
    embedded_right[:, 0] =embedded_right[:, 0]
    embedded_right[:, 1] =embedded_right[:, 1] - 40
    embedded[index_right] = embedded_right
    return embedded


def get_cylinder(vertice):
    x, y = vertice[:,0],vertice[:,1]
    z_min, z_max = np.min(vertice[:,2]), np.max(vertice[:,2])
    center, r =  compute_least_squaure_circle(x, y)
    return (center,r), (z_min, z_max)

def get_2d_embedded(vertice,faces, down_sample_index,ninter, is_fc=True):
    if is_fc:
        circle, z_range = get_cylinder(vertice)
        embedded,plot_xy = get_projection_from_circle_and_vertice(vertice, circle)
        em_weight = compute_embeded_weight(vertice,faces,embedded,ninter,ratio_max=80)
        vertice_cylinder = extract_suface_vertice_from_cylinder(circle,len(down_sample_index),z_range)
        visualize_surface_and_cylinder(vertice[down_sample_index],vertice_cylinder)
    else:
        vertice_left = vertice[vertice[:,0]<50]
        index_left = np.where(vertice[:,0]<50)[0]
        vertice_right = vertice[vertice[:,0]>=50]
        index_right = np.where(vertice[:, 0] >= 50)[0]
        embedded_left = do_linear_pca(vertice_left)
        embedded_right = do_linear_pca(vertice_right)
        embedded_left = rotate_embedded(embedded_left, -50)
        embedded_right = rotate_embedded(embedded_right, -160)
        embedded_right[:,0] = - embedded_right[:,0] # flip x
        do_visualization_on_3d(vertice,sample_num=3500, bias=0,factor=1,c=vertice[:,2])
        #do_visualization_on_3d(vertice_left,sample_num=3500, bias=0,factor=1,c=vertice_left[:3500,2])
        # do_visualization_on_3d(vertice_right,sample_num=3500, bias=0,factor=1,c=vertice_right[:3500,2])
        do_visualization_colorized_on_2d(embedded_left,[embedded_left[:3500,0],embedded_left[:3500,1],vertice_left[:3500,2]])
        do_visualization_colorized_on_2d(embedded_right,[embedded_right[:3500,0],embedded_right[:3500,1],vertice_right[:3500,2]])
        # do_visualization_embedded_on_3d(embedded_left,c=embedded_left[:,0],sample_num=3500,  bias=0, factor=1)
        # do_visualization_embedded_on_3d(embedded_right,c=embedded_right[:,0],sample_num=3500,  bias=0, factor=1)
        embedded = get_combined_TC_embedding(embedded_left, embedded_right,index_left,index_right)
        em_weight = compute_embeded_weight(vertice, faces, embedded, ninter,ratio_max=5)
        plot_xy = embedded
        do_visualization_colorized_on_2d(embedded[down_sample_index])

    return embedded,em_weight,plot_xy







fn_vertice_list = ['data/FC_inner_vertice.npy','data/TC_inner_vertice.npy']
fn_thick_list = ['data/FC_inner_thickness.npy','data/TC_inner_thickness.npy']
fn_faces_list = ['data/FC_inner_faces.npy','data/TC_inner_faces.npy']
fn_embedded_list = ['output/FC_inner_embedded.npy','output/TC_inner_embedded.npy']
downsampled_ratio = [1.0,1.0]
visual_downsampled_ratio=[0.1,0.1]
do_mds = False
do_visual = True
min_adj =0.1
scale_z_factor = 1.4
angle = 90
ninter = 300

for i in range(len(fn_vertice_list)):
    is_fc = i==0
    vertice =np.load(fn_vertice_list[i])
    faces = np.load(fn_faces_list[i])
    vertice = rotate_z(vertice, angle, is_fc = is_fc)
    downsampled_id, _ = do_downsampling(vertice, visual_downsampled_ratio[i])
    embedded,em_weight,plot_xy = get_2d_embedded(vertice,faces, downsampled_id, ninter, is_fc=is_fc)
    np.save(fn_embedded_list[i], embedded)
    if do_visual:
        embedded = np.load(fn_embedded_list[i])
        vertice = np.load(fn_vertice_list[i])
        downsampled_id, vertice = do_downsampling(vertice, visual_downsampled_ratio[i])
        thickness = np.load(fn_thick_list[i])
        #thickness[thickness<min_adj]=0.
        #do_visualization_on_3d(vertice,sample_num=1000, bias=2500,factor=1)
        #do_visualization_embedded_on_3d(embedded[downsampled_id],sample_num=1000, bias=2500,factor=1)
        #do_visualization_colorized_on_2d(embedded[downsampled_id])
        #do_visualization_on_2d_thickness(embedded[downsampled_id], thickness[downsampled_id])
        __map_thickness_to_2D_projection(plot_xy, thickness,ninter=300,fpth=None)
        __map_thickness_to_2D_projection(plot_xy, em_weight,ninter=300,fpth=None)
        #do_grid_interpolation(embedded,thickness,ninter=300)
