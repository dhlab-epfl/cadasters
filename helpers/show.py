import numpy as np
import cv2
from skimage.segmentation import mark_boundaries


def show_superpixels(image, segments, namefile):
    """
    Save superpixels visualization in file

    :param image: image to be drawn
    :param segments: superpixels to draw
    :param namefile: name of the file to save the image
    :return:
    """
    imsuperpix = mark_boundaries(image, segments, color=(1, 1, 1), outline_color=(0,0,0), mode='thick')
    imsuperpix = np.uint8(imsuperpix*255)
    cv2.imwrite(namefile, imsuperpix)
# -----------------------------------------------------------


def show_class(merged_segments, graph, namefile):
    """
    Shows which superpixels/regions belong to which class and saves the plot

    :param merged_segments: map of merged superpixels /regions
    :param graph: graph of the image
    :param namefile: name of the file to save the plot
    """

    pred_seg = np.zeros([merged_segments.shape[0], merged_segments.shape[1], 3])
    Rd = 255*np.ones(merged_segments.shape)
    Gn = 255*np.ones(merged_segments.shape)
    Be = 255*np.ones(merged_segments.shape)
    for s in graph.nodes():
        pixLbl = merged_segments == s
        if 'class' in graph.node[s] and graph.node[s]['class'] == 0:
            Rd[pixLbl] = 255
            Gn[pixLbl] = 0
            Be[pixLbl] = 0
        elif 'class' in graph.node[s] and graph.node[s]['class'] == 1:
            Rd[pixLbl] = 0
            Gn[pixLbl] = 0
            Be[pixLbl] = 0
        elif 'class' in graph.node[s] and graph.node[s]['class'] == 2:
            Rd[pixLbl] = 255
            Gn[pixLbl] = 222
            Be[pixLbl] = 0

    pred_seg[:, :, 0] = Be
    pred_seg[:, :, 1] = Gn
    pred_seg[:, :, 2] = Rd

    cv2.imwrite(namefile, pred_seg)
# -----------------------------------------------------------------------


def show_polygons(image, dic_polygon, color, filename):
    """
    Draws polygons in the given image and saves it

    :param image: image to plot parcels
    :param dic_polygon: dictionary containing the coordinates of all polygons. tuple (uuid,polygon(s))
    :param color: color to draw polygon
    :param filename: filename of the image to save
    """
    img_pol = image.copy()
    for nod, list_tup in dic_polygon.items():
        for tup in list_tup:
            cv2.fillPoly(img_pol, tup[1], color)

    cv2.imwrite(filename, img_pol)
# ------------------------------------------------------------------------


def show_boxes(image, listBox, color, filename=None):
    """
    Draws the boxes around text in image

    :param image: Image to be drawn
    :param listBox: list of Box object to be drawn
    :param color: color of the box
    :param filename: name of the file to save drawn image
    :return:
    """

    for box in listBox:
        cv2.drawContours(image, [box.box_pts], 0, color)
        # cv2.putText(image, str(box.lbl_polygon),tuple(box.box_pts[0]),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0))
    if filename:
        cv2.imwrite(filename, image)
# -------------------------------------------------------------------------


def plot_graph_edges(img, segments, graph, centers, namefile):
    imsuperpix = mark_boundaries(img, segments, color=(1, 1, 1), outline_color=(0,0,0), mode='thick')
    imsuperpix = np.uint8(imsuperpix*255)
    for edge in graph.edges():
        cv2.line(imsuperpix, (np.uint64(centers[edge[0]][0]),np.uint64(centers[edge[0]][1])),
                (np.uint64(centers[edge[1]][0]), np.uint64(centers[edge[1]][1])) , (255,0,0))

    cv2.imwrite(namefile, imsuperpix)

    return
# --------------------------------------------------------------------------


def show_orientation(img2draw, eigvect, center, filename=None):

    diagonal = eigvect[0]
    linex = int(center[0] + diagonal[0]*20)
    liney = int(center[0] + diagonal[1]*20)
    cv2.circle(img2draw, (int(center[0]), int(center[1])), 1, 128, -1)
    cv2.line(img2draw, (int(center[0]), int(center[1])), (linex, liney), 128, 1)

    if filename:
        cv2.imwrite(filename, img2draw)
