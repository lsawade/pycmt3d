#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for source inversion and grid search. Combines cmt3d and grid3d

:copyright:
    Lucas Sawade (lsawade@princeton.edu), 2020
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

Last Update: January 2020

"""

import os
import numpy as np
from owslib.wfs import WebFeatureService
from lxml import etree
from collections import Counter
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from cartopy.crs import PlateCarree
import cartopy


def sample_xml(filename, opts):
    """Returns file rad with options as string."""
    with open(filename, opts) as xml:
        return xml.read()


def print_xml_structure(filename):
    """Prints structure of a given XML file."""

    xml_as_bytes = sample_xml(filename, 'rb')
    root = etree.fromstring(xml_as_bytes)

    tree = etree.ElementTree(root)

    print_substructure(root, tree)


def print_substructure(root, tree):
    """Prints tree structure"""

    for tag in root.iter():
        path = tree.getpath(tag)
        path = path.replace('/', '    ')
        spaces = Counter(path)
        tag_name = path.split()[-1].split('[')[0]
        tag_name = ' ' * (spaces[' '] - 4) + tag_name
        print(tag_name)


def print_full_tree(tree):
    """Prints XML tree structure."""
    print(tree.tostring(tree, pretty_print=True))


def download_wfs():

    wfs11 = WebFeatureService(url='http://mapsref.brgm.fr/wxs/1GG/'
                                  'CGMW_Bedrock_and_Structural_Geology',
                              version='2.0.0')
    print(wfs11.identification.type)
    print(wfs11.identification.title)

    print([operation.name for operation in wfs11.operations])

    print([wfs11.contents])

    for typename, meta in wfs11.contents.items():
        print(typename, meta)
        response = wfs11.getfeature(typename=typename)
        out = open(typename[3:]+'.gml', 'wb')
        out.write(response.read())
        out.close()


class Member(object):
    def __init__(self, coordinates: list = None, bbox: np.ndarray = None,
                 description: str = None, lithology: str = None,
                 stratigraphy: str = None):
        """
        One member of a feature map and its parameters.

        :param coordinates: list of polygon coordinate sets
        :param bbox: bounding box
        :param description: description
        :param lithology: lithologies
        :param stratigraphy: stratigraphies
        """

        self.coordinates = coordinates
        self.bbox = bbox
        self.description = description
        self.lithology = lithology
        self.stratigraphy = stratigraphy

    @classmethod
    def from_dict(cls, d: dict):
        """Creates Member from member dictionary"""
        return cls(coordinates=d["coordinates"],
                   description=d["description"],
                   lithology=d["lithology"],
                   stratigraphy=d["stratigraphy"],
                   bbox=d["bbox"])

    @property
    def npoly(self):
        """Returns the number of coordinates that describe the polygon."""
        return len(self.coordinates)

    def __len__(self):
        return len(self.coordinates)

    def __str__(self):
        string = "\n  The member contains a geological feature with\n" \
                 "  following:\n\n"
        string += "    Lithology: %s\n" % self.lithology
        string += "    Stratigraphy: %s\n" % self.stratigraphy
        string += "    Description: %s\n\n" % self.description
        string += "  and extent:\n"
        string += "    Lower Corner: (%4.3f, %4.3f)\n" % tuple(self.bbox[0:2])
        string += "    Upper Corner: (%4.3f, %4.3f)\n\n" % tuple(self.bbox[2:])

        return string


class GeoMap(object):

    def __init__(self, members: list = None, descriptions: set = None,
                 lithologies: set = None, stratigraphies: set = None):
        """
        Contains all features to of a geological map loaded from an xml.

        :param members: list of Members
        :param descriptions: set of descriptions
        :param lithologies: set of lithologies
        :param stratigraphies: set of stratigraphic data
        """
        self.members = members
        self.descriptions = descriptions
        self.lithologies = lithologies
        self.stratigraphies = stratigraphies

    def add(self, member: Member):
        if self.lithologies is None:
            self.lithologies = set()
        if self.descriptions is None:
            self.descriptions = set()
        if self.stratigraphies is None:
            self.stratigraphies = set()
        if self.members is None:
            self.members = []

        self.members.append(member)
        self.lithologies.add(member.lithology)
        self.stratigraphies.add(member.stratigraphy)
        self.descriptions.add(member.description)

    def __add__(self, other):

        if self.lithologies is None:
            self.lithologies = set()
        if self.descriptions is None:
            self.descriptions = set()
        if self.stratigraphies is None:
            self.stratigraphies = set()
        if self.members is None:
            self.members = []

        self.members.extend(other.members)
        self.lithologies.update(other.lithologies)
        self.stratigraphies.update(other.stratigraphies)
        self.descriptions.update(other.descriptions)

        return self

    @property
    def nmembers(self):
        return len(self.members)

    @property
    def nlith(self):
        """Returns the number of different lithologies in Geomap"""
        return len(self.lithologies)

    @property
    def nstrat(self):
        """Returns the number of different stratigraphies in Geomap"""
        return len(self.stratigraphies)

    @property
    def ndescr(self):
        """Returns the number of different descriptions in Geomap"""
        return len(self.descriptions)

    def save_json(self, outfileName: str = "outfile.json"):
        """
        Saves the dictionary d to the outfileName file.
        :param d: dictionary
        :param outfileName:
        :return:
        """
        d = dict()

        d["members"] = self.members
        for j, member in enumerate(d["members"]):
            for k, coordinates in enumerate(member.coordinates):
                d["members"][j].coordinates[k] = coordinates.tolist()
            d["members"][j].bbox = d["members"][j].bbox.tolist()
        d["members"] = [x.__dict__ for x in self.members]
        d["lithologies"] = list(self.lithologies)
        d["stratigraphies"] = list(self.stratigraphies)
        d["descriptions"] = list(self.descriptions)
        d["readme"] = self.__str__()

        with open(outfileName, 'w') as file:
            json.dump(d, file)

    @classmethod
    def load_json(cls, filename):
        with open(filename, 'r') as file:
            d = json.load(file)

        members = []
        for j, member in enumerate(d["members"]):
            members.append(Member(coordinates=[np.array(x) for x in
                                               member["coordinates"]],
                                  description=member["description"],
                                  lithology=member["lithology"],
                                  stratigraphy=member["stratigraphy"],
                                  bbox=member["bbox"]))
        descriptions = set(d["descriptions"])
        stratigraphies = set(d["stratigraphies"])
        lithologies = set(d["lithologies"])

        return cls(members=members,  descriptions=descriptions,
                   lithologies=lithologies, stratigraphies=stratigraphies)

    @classmethod
    def from_xml(cls, filename):
        """Loads content from a GML xml file."""

        # Get namespaces
        ns = get_xml_namespaces(filename)

        # Open file in byte format for faster reading
        xml_as_bytes = sample_xml(filename, 'rb')
        root = etree.fromstring(xml_as_bytes)

        # Set namespaces
        xmlns_ms = ns["ms"]
        xmlns_gml = ns["gml"]
        # xmlns_wfs = ns["wfs"]
        # xmlns_xsi = ns["xsi"]

        members = []
        counter = 0

        for k, element in enumerate(root.findall('wfs:member',
                                                 namespaces=root.nsmap)):

            # Create empty
            member = Member()

            for subelement in element.iter(xmlns_gml + "lowerCorner",
                                           xmlns_gml + "upperCorner",
                                           xmlns_gml + "posList",
                                           xmlns_ms + "LITHO_EN",
                                           xmlns_ms + "STRATI_EN",
                                           xmlns_ms + "DESCR_EN", ):

                if "posList" in subelement.tag:
                    if member.coordinates is None:
                        counter = 0
                        member.coordinates = []

                    # Converting the text elements in the member into
                    # float arrays
                    member.coordinates.append(np.array([
                        float(x) for x in subelement.text.strip().split(' ')]))
                    member.coordinates[counter] = \
                        member.coordinates[counter] \
                        .reshape((int(member.coordinates[counter].size / 2),
                                  2))
                    counter += 1

                elif "lowerCorner" in subelement.tag:
                    lowercorner = [float(x)
                                   for x in subelement.text.strip().split(' ')]
                elif "upperCorner" in subelement.tag:
                    uppercorner = [float(x)
                                   for x in subelement.text.strip().split(' ')]
                elif "LITHO_EN" in subelement.tag:
                    if subelement.text is None:
                        member.lithology = "Not classifed."
                    else:
                        member.lithology = subelement.text
                elif "STRATI_EN" in subelement.tag:
                    if subelement.text is None:
                        member.stratigraphy = "Not classifed."
                    else:
                        member.stratigraphy = subelement.text
                elif "DESCR_EN" in subelement.tag:
                    if subelement.text is None:
                        member.description = "Not classifed."
                    else:
                        member.description = subelement.text

            # Create bounding box from upper and lower Corner
            member.bbox = []
            member.bbox.extend(lowercorner)
            member.bbox.extend(uppercorner)
            member.bbox = np.array(member.bbox)

            members.append(member)

        stratigraphies = set()
        lithologies = set()
        descriptions = set()

        for member in members:
            stratigraphies.add(member.stratigraphy)
            lithologies.add(member.lithology)
            descriptions.add(member.description)

        return cls(members=members,  descriptions=descriptions,
                   lithologies=lithologies, stratigraphies=stratigraphies)

    def __str__(self):
        readme = "GeoMap:\n\n"
        readme += "  Created on " + datetime.now().strftime(
            '%Y-%m-%dT%H:%M:%SZ')
        readme += "\n\n"
        readme += "  The geomap dictionary contains information about the \n" \
                  "  lithology, stratigraphy, and a combining descriptor,\n" \
                  "  which can be used as legends. The main convenience of\n" \
                  "  this format is the included list of polygons which \n" \
                  "  each contain the aforementioned descriptors, \n" \
                  "  lithological, and stratigraphical information needed \n" \
                  "  to be used as legends Each Polygon contains info for\n" \
                  "  one single geological region/feature.\n\n" \
                  "  Features included in his map:\n\n"
        for description in self.descriptions:
            readme += "  o  %s\n" % description

        return readme


def get_xml_namespaces(filename: str):
    """
    Extracts namespaces from XML files into a dictionary

    :param filename: filename
    :return: dict
    """
    namepaces = dict()
    with open(filename, 'r') as xml:
        for line in xml:
            if "xmlns:" in line:
                newline = line.split("xmlns:")[1].strip().split("=")
                namepaces[newline[0]] = \
                    "{" + newline[1].strip().strip('"') + "}"

    return namepaces


if __name__ == "__main__":

    fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "geology", "geology.json")
    geomap = GeoMap.load_json(fname)

    fig = plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
    g = gridspec.GridSpec(5, 2)
    ax = plt.axes(projection=PlateCarree(0.0))

    ax.frameon = True
    ax.outline_patch.set_linewidth(0.75)
    # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
    # function around 180deg
    gl = ax.gridlines(crs=PlateCarree(), draw_labels=False,
                      linewidth=1, color='lightgray', alpha=0.5,
                      linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = True
    # ax.outline_patch.set_visible(False)

    # Change fontsize
    fontsize = 12
    font_dict = {"fontsize": fontsize,
                 "weight": "bold"}
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=font_dict)
    ax.set_yticklabels(ax.get_yticklabels(), fontdict=font_dict)

    ax.add_feature(cartopy.feature.LAND, zorder=200, lw=2,
                   edgecolor='black', facecolor="none")

    # Get discrete colormap
    cmap = cm.get_cmap('gist_ncar', geomap.ndescr)

    colordict = dict()
    for k, descriptions in enumerate(sorted(list(geomap.descriptions))):
        colordict[descriptions] = k

    patches = []
    colors = []
    for member in geomap.members:

        for poly in member.coordinates:
            patches.append(Polygon(np.fliplr(poly), joinstyle="round",
                                   fill=True, edgecolor=None))
            colors.append(colordict[member.description])

    p = PatchCollection(patches, cmap=cmap, alpha=0.4, zorder=100)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    ax.set_global()

    padding = 5.
    cmt_lon = -119.
    cmt_lat = 36.
    minlon = cmt_lon - padding
    maxlon = cmt_lon + padding
    minlat = cmt_lat - padding
    maxlat = cmt_lat + padding

    # Updated parallels..
    ax.set_extent([minlon - padding, maxlon + padding,
                   minlat - padding, maxlat + padding])

    plt.figure(figsize=(15, 8), facecolor='w', edgecolor='k')
    ax1 = plt.axes()
    circs = []
    labels = []

    for key, value in colordict.items():
        circs.append(Line2D([0], [0], linestyle="none", marker="s", alpha=0.4,
                            markersize=10, markerfacecolor=cmap(value)))
        labels.append(key)

    ax1.legend(circs, labels, numpoints=1,
               loc="best", ncol=2, fontsize=8, frameon=False)
    ax1.axis('off')

    plt.show()
