# -*- coding: utf-8 -*-
"""
c50py._export
=============

Tree visualisation for :class:`~c50py.C5Classifier` and
:class:`~c50py.C5Regressor` with the *same* look, parameters and output as
scikit-learn's :func:`sklearn.tree.plot_tree`, :func:`sklearn.tree.export_graphviz`
and :func:`sklearn.tree.export_text`.

The drawing code (colour palette, node text, Reingold-Tilford layout,
matplotlib annotations and DOT output) is adapted from scikit-learn
(``sklearn/tree/_export.py`` and ``sklearn/tree/_reingold_tilford.py``,
BSD-3-Clause, Copyright (c) the scikit-learn developers).  It is vendored here
instead of importing scikit-learn's private classes so that the figures stay
identical across scikit-learn versions.

Differences that come from the C5.0 model itself, not from the drawing:

* The impurity shown for classifiers is the **entropy** (bits), the measure
  C5.0 uses to choose splits.  For regressors it is the ``squared_error``
  (weighted variance) of the node, exactly as in scikit-learn.
* ``samples`` is the (weighted) number of training cases that reached the
  node.  C5.0 sends cases with a missing value down both branches with
  fractional weights, so it can be non-integer when there are missing values.
* Categorical splits are written ``feature in {a, b}``; the ``True`` branch
  (left) holds the listed categories.
"""
from __future__ import annotations

from collections.abc import Iterable
from io import StringIO
from numbers import Integral

import numpy as np

__all__ = ["plot_tree", "export_graphviz", "export_text"]

TREE_LEAF = -1
TREE_UNDEFINED = -2


# -----------------------------------------------------------------------------
# Flat (array) representation of a C5 tree, shaped like ``sklearn.tree._tree.Tree``
# -----------------------------------------------------------------------------
class _FlatTree:
    """Array view of a c50py tree with the attributes the exporters need.

    Nodes are numbered in depth-first pre-order (root = 0, then the left
    subtree, then the right one), which is the same numbering scikit-learn uses.
    """

    def __init__(self):
        self.children_left = []
        self.children_right = []
        self.feature = []
        self.threshold = []
        self.split_type = []      # "numeric" | "categorical" | None
        self.categories = []      # sorted list for categorical splits
        self.impurity = []
        self.n_node_samples = []
        self.weighted_n_node_samples = []
        self.value = []
        self.n_outputs = 1
        self.n_classes = [1]
        self.n_features = 0

    def _finalise(self):
        self.children_left = np.asarray(self.children_left, dtype=np.intp)
        self.children_right = np.asarray(self.children_right, dtype=np.intp)
        self.feature = np.asarray(self.feature, dtype=np.intp)
        self.impurity = np.asarray(self.impurity, dtype=float)
        self.n_node_samples = np.asarray(self.n_node_samples, dtype=float)
        self.weighted_n_node_samples = np.asarray(self.weighted_n_node_samples, dtype=float)
        self.value = np.asarray(self.value, dtype=float)
        self.node_count = len(self.children_left)
        self.n_classes = np.asarray(self.n_classes, dtype=np.intp)
        return self


def _entropy_bits(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum()) + 0.0   # + 0.0 turns -0.0 into 0.0


def _is_classifier(model) -> bool:
    return getattr(model, "_estimator_type", None) == "classifier" or hasattr(model, "classes_")


def _get_root(model, tree_index=0):
    ens = getattr(model, "ensemble_", None)
    if getattr(model, "tree_", None) is None and not ens:
        raise ValueError("Estimator not fitted. Call fit(...) first.")
    if ens and (len(ens) > 1 or getattr(model, "tree_", None) is None):
        if not (0 <= int(tree_index) < len(ens)):
            raise ValueError(f"tree_index must be in [0, {len(ens) - 1}], got {tree_index}")
        return ens[int(tree_index)]
    if tree_index not in (0, None):
        raise ValueError("tree_index > 0 only makes sense for boosted models (trials > 1)")
    return model.tree_


def _n_features(model):
    for attr in ("n_features_", "n_features_in_"):
        n = getattr(model, attr, None)
        if n is not None:
            return int(n)
    is_cat = getattr(model, "is_cat_", None)
    return len(is_cat) if is_cat is not None else None


def _flatten(model, tree_index=0) -> _FlatTree:
    root = _get_root(model, tree_index)
    ft = _FlatTree()
    ft.n_features = _n_features(model) or 0
    clf = _is_classifier(model)
    if clf:
        classes = list(model.classes_)
        ft.n_classes = [len(classes)]
    else:
        ft.n_classes = [1]

    def add(node):
        nid = len(ft.children_left)
        ft.children_left.append(TREE_LEAF)
        ft.children_right.append(TREE_LEAF)
        is_leaf = bool(node.is_leaf) or not node.children
        if is_leaf:
            ft.feature.append(TREE_UNDEFINED)
            ft.threshold.append(TREE_UNDEFINED)
            ft.split_type.append(None)
            ft.categories.append(None)
        else:
            ft.feature.append(int(node.feature_index))
            if node.split_type == "numeric":
                ft.threshold.append(float(node.threshold))
                ft.split_type.append("numeric")
                ft.categories.append(None)
            else:
                ft.threshold.append(TREE_UNDEFINED)
                ft.split_type.append("categorical")
                try:
                    cats = sorted(node.threshold)
                except TypeError:
                    cats = sorted(node.threshold, key=str)
                ft.categories.append(cats)

        if clf:
            dist = node.class_distribution or {}
            counts = np.array([float(dist.get(c, 0.0)) for c in classes])
            tot = counts.sum()
            ft.weighted_n_node_samples.append(tot)
            ft.n_node_samples.append(tot)
            ft.impurity.append(_entropy_bits(counts))
            frac = counts / tot if tot > 0 else counts
            ft.value.append(frac.reshape(1, -1))
        else:
            n = float(node.n_samples)
            ft.weighted_n_node_samples.append(n)
            ft.n_node_samples.append(n)
            ft.impurity.append(float(node.sse) / n if n > 0 else 0.0)
            ft.value.append(np.array([[float(node.predicted_value)]]))

        if not is_leaf:
            ft.children_left[nid] = add(node.children["left"])
            ft.children_right[nid] = add(node.children["right"])
        return nid

    add(root)
    return ft._finalise()


def _default_feature_names(model, feature_names):
    """Names given explicitly win; otherwise use the names seen in ``fit``.

    The placeholder names ``f0, f1, ...`` that c50py generates when no names
    are known are treated as "no names", so the plot falls back to
    scikit-learn's ``x[0], x[1], ...``.
    """
    if feature_names is not None:
        return list(feature_names)
    fn = getattr(model, "feature_names_", None)
    if fn is None:
        return None
    fn = list(fn)
    if fn == [f"f{i}" for i in range(len(fn))]:
        return None
    return fn


def _check_feature_names(feature_names, ft):
    if feature_names is not None and ft.n_features and len(feature_names) != ft.n_features:
        raise ValueError(
            "Length of feature_names, %d does not match number of features, %d"
            % (len(feature_names), ft.n_features)
        )


def _criterion(model):
    return "entropy" if _is_classifier(model) else "squared_error"


def _fmt_count(n, precision):
    if float(n).is_integer():
        return str(int(n))
    return str(round(float(n), precision))


# -----------------------------------------------------------------------------
# Exporters (adapted from scikit-learn, BSD-3-Clause)
# -----------------------------------------------------------------------------
def _color_brew(n):
    """Generate n colors with equally spaced hues (same palette as scikit-learn)."""
    color_list = []
    s, v = 0.75, 0.9
    c = s * v
    m = v - c
    for h in np.arange(25, 385, 360.0 / n).astype(int):
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        rgb = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x), (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)
    return color_list


class _BaseTreeExporter:
    def __init__(self, max_depth=None, feature_names=None, class_names=None,
                 label="all", filled=False, impurity=True, node_ids=False,
                 proportion=False, rounded=False, precision=3, fontsize=None):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        if self.colors["bounds"] is None:
            color = list(self.colors["rgb"][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0.0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            color = list(self.colors["rgb"][0])
            alpha = (value - self.colors["bounds"][0]) / (
                self.colors["bounds"][1] - self.colors["bounds"][0]
            )
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        return "#%2x%2x%2x" % tuple(color)

    def get_fill_color(self, tree, node_id):
        if "rgb" not in self.colors:
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        node_val = tree.value[node_id][0, :]
        if tree.n_classes[0] == 1 and isinstance(node_val, Iterable) and self.colors["bounds"] is not None:
            node_val = node_val.item()
        return self.get_color(node_val)

    def _split_text(self, tree, node_id):
        characters = self.characters
        if self.feature_names is not None:
            feature = self.str_escape(str(self.feature_names[tree.feature[node_id]]))
        else:
            feature = "x%s%s%s" % (characters[1], tree.feature[node_id], characters[2])
        if tree.split_type[node_id] == "categorical":
            cats = ", ".join(self.str_escape(str(c)) for c in tree.categories[node_id])
            return "%s %s {%s}%s" % (feature, self.in_symbol, cats, characters[4])
        return "%s %s %s%s" % (feature, characters[3],
                               round(tree.threshold[node_id], self.precision), characters[4])

    def node_to_str(self, tree, node_id, criterion):
        value = tree.value[node_id][0, :]
        labels = (self.label == "root" and node_id == 0) or self.label == "all"
        characters = self.characters
        node_string = characters[-1]

        if self.node_ids:
            if labels:
                node_string += "node "
            node_string += characters[0] + str(node_id) + characters[4]

        if tree.children_left[node_id] != TREE_LEAF:
            node_string += self._split_text(tree, node_id)

        if self.impurity:
            if labels:
                node_string += "%s = " % criterion
            node_string += str(round(tree.impurity[node_id], self.precision)) + characters[4]

        if labels:
            node_string += "samples = "
        if self.proportion:
            percent = 100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
            node_string += str(round(percent, 1)) + "%" + characters[4]
        else:
            node_string += _fmt_count(tree.n_node_samples[node_id], self.precision) + characters[4]

        if not self.proportion and tree.n_classes[0] != 1:
            value = value * tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += "value = "
        if tree.n_classes[0] == 1:
            value_text = np.around(value, self.precision)
        elif self.proportion:
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            value_text = value.astype(int)
        else:
            value_text = np.around(value, self.precision)
        value_text = str(value_text.astype("S32")).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        if self.class_names is not None and tree.n_classes[0] != 1:
            if labels:
                node_string += "class = "
            if self.class_names is not True:
                class_name = self.str_escape(str(self.class_names[np.argmax(value)]))
            else:
                class_name = "y%s%s%s" % (characters[1], np.argmax(value), characters[2])
            node_string += class_name

        if node_string.endswith(characters[4]):
            node_string = node_string[: -len(characters[4])]
        return node_string + characters[5]

    def str_escape(self, string):
        return string


class _DOTTreeExporter(_BaseTreeExporter):
    def __init__(self, out_file=None, max_depth=None, feature_names=None,
                 class_names=None, label="all", filled=False, leaves_parallel=False,
                 impurity=True, node_ids=False, proportion=False, rotate=False,
                 rounded=False, special_characters=False, precision=3,
                 fontname="helvetica"):
        super().__init__(max_depth=max_depth, feature_names=feature_names,
                         class_names=class_names, label=label, filled=filled,
                         impurity=impurity, node_ids=node_ids, proportion=proportion,
                         rounded=rounded, precision=precision)
        self.leaves_parallel = leaves_parallel
        self.out_file = out_file
        self.special_characters = special_characters
        self.fontname = fontname
        self.rotate = rotate
        if special_characters:
            self.characters = ["&#35;", "<SUB>", "</SUB>", "&le;", "<br/>", ">", "<"]
            self.in_symbol = "&isin;"
        else:
            self.characters = ["#", "[", "]", "<=", "\\n", '"', '"']
            self.in_symbol = "in"
        self.ranks = {"leaves": []}
        self.colors = {"bounds": None}

    def export(self, tree, criterion):
        self.head()
        self.recurse(tree, 0, criterion=criterion)
        self.tail()

    def tail(self):
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write("{rank=same ; " + "; ".join(r for r in self.ranks[rank]) + "} ;\n")
        self.out_file.write("}")

    def head(self):
        self.out_file.write("digraph Tree {\n")
        self.out_file.write("node [shape=box")
        rounded_filled = []
        if self.filled:
            rounded_filled.append("filled")
        if self.rounded:
            rounded_filled.append("rounded")
        if len(rounded_filled) > 0:
            self.out_file.write(', style="%s", color="black"' % ", ".join(rounded_filled))
        self.out_file.write(', fontname="%s"' % self.fontname)
        self.out_file.write("] ;\n")
        if self.leaves_parallel:
            self.out_file.write("graph [ranksep=equally, splines=polyline] ;\n")
        self.out_file.write('edge [fontname="%s"] ;\n' % self.fontname)
        if self.rotate:
            self.out_file.write("rankdir=LR ;\n")

    def recurse(self, tree, node_id, criterion, parent=None, depth=0):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if self.max_depth is None or depth <= self.max_depth:
            if left_child == TREE_LEAF:
                self.ranks["leaves"].append(str(node_id))
            elif str(depth) not in self.ranks:
                self.ranks[str(depth)] = [str(node_id)]
            else:
                self.ranks[str(depth)].append(str(node_id))
            self.out_file.write("%d [label=%s" % (node_id, self.node_to_str(tree, node_id, criterion)))
            if self.filled:
                self.out_file.write(', fillcolor="%s"' % self.get_fill_color(tree, node_id))
            self.out_file.write("] ;\n")
            if parent is not None:
                self.out_file.write("%d -> %d" % (parent, node_id))
                if parent == 0:
                    angles = np.array([45, -45]) * ((self.rotate - 0.5) * -2)
                    self.out_file.write(" [labeldistance=2.5, labelangle=")
                    if node_id == 1:
                        self.out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        self.out_file.write('%d, headlabel="False"]' % angles[1])
                self.out_file.write(" ;\n")
            if left_child != TREE_LEAF:
                self.recurse(tree, left_child, criterion=criterion, parent=node_id, depth=depth + 1)
                self.recurse(tree, right_child, criterion=criterion, parent=node_id, depth=depth + 1)
        else:
            self.ranks["leaves"].append(str(node_id))
            self.out_file.write('%d [label="(...)"' % node_id)
            if self.filled:
                self.out_file.write(', fillcolor="#C0C0C0"')
            self.out_file.write("] ;\n")
            if parent is not None:
                self.out_file.write("%d -> %d ;\n" % (parent, node_id))

    def str_escape(self, string):
        return string.replace('"', r"\"")


# --- Reingold-Tilford / Buchheim layout (scikit-learn, BSD-3-Clause) ----------
class _RTTree:
    def __init__(self, label="", node_id=-1, *children):
        self.label = label
        self.node_id = node_id
        self.children = children if children else []


class _DrawTree:
    def __init__(self, tree, parent=None, depth=0, number=1):
        self.x = -1.0
        self.y = depth
        self.tree = tree
        self.children = [_DrawTree(c, self, depth + 1, i + 1) for i, c in enumerate(tree.children)]
        self.parent = parent
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        self.number = number

    def left(self):
        return self.thread or (len(self.children) and self.children[0])

    def right(self):
        return self.thread or (len(self.children) and self.children[-1])

    def lbrother(self):
        n = None
        if self.parent:
            for node in self.parent.children:
                if node == self:
                    return n
                n = node
        return n

    def get_lmost_sibling(self):
        if not self._lmost_sibling and self.parent and self != self.parent.children[0]:
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling

    lmost_sibling = property(get_lmost_sibling)

    def max_extents(self):
        extents = [c.max_extents() for c in self.children]
        extents.append((self.x, self.y))
        return np.max(extents, axis=0)


def _buchheim(tree):
    dt = _first_walk(_DrawTree(tree))
    mn = _second_walk(dt)
    if mn < 0:
        _third_walk(dt, -mn)
    return dt


def _third_walk(tree, n):
    tree.x += n
    for c in tree.children:
        _third_walk(c, n)


def _first_walk(v, distance=1.0):
    if len(v.children) == 0:
        v.x = v.lbrother().x + distance if v.lmost_sibling else 0.0
    else:
        default_ancestor = v.children[0]
        for w in v.children:
            _first_walk(w)
            default_ancestor = _apportion(w, default_ancestor, distance)
        _execute_shifts(v)
        midpoint = (v.children[0].x + v.children[-1].x) / 2
        w = v.lbrother()
        if w:
            v.x = w.x + distance
            v.mod = v.x - midpoint
        else:
            v.x = midpoint
    return v


def _apportion(v, default_ancestor, distance):
    w = v.lbrother()
    if w is not None:
        vir = vor = v
        vil = w
        vol = v.lmost_sibling
        sir = sor = v.mod
        sil = vil.mod
        sol = vol.mod
        while vil.right() and vir.left():
            vil = vil.right()
            vir = vir.left()
            vol = vol.left()
            vor = vor.right()
            vor.ancestor = v
            shift = (vil.x + sil) - (vir.x + sir) + distance
            if shift > 0:
                _move_subtree(_ancestor(vil, v, default_ancestor), v, shift)
                sir = sir + shift
                sor = sor + shift
            sil += vil.mod
            sir += vir.mod
            sol += vol.mod
            sor += vor.mod
        if vil.right() and not vor.right():
            vor.thread = vil.right()
            vor.mod += sil - sor
        else:
            if vir.left() and not vol.left():
                vol.thread = vir.left()
                vol.mod += sir - sol
            default_ancestor = v
    return default_ancestor


def _move_subtree(wl, wr, shift):
    subtrees = wr.number - wl.number
    wr.change -= shift / subtrees
    wr.shift += shift
    wl.change += shift / subtrees
    wr.x += shift
    wr.mod += shift


def _execute_shifts(v):
    shift = change = 0
    for w in v.children[::-1]:
        w.x += shift
        w.mod += shift
        change += w.change
        shift += w.shift + change


def _ancestor(vil, v, default_ancestor):
    if vil.ancestor in v.parent.children:
        return vil.ancestor
    return default_ancestor


def _second_walk(v, m=0, depth=0, mn=None):
    v.x += m
    v.y = depth
    if mn is None or v.x < mn:
        mn = v.x
    for w in v.children:
        mn = _second_walk(w, m + v.mod, depth + 1, mn)
    return mn


class _MPLTreeExporter(_BaseTreeExporter):
    def __init__(self, max_depth=None, feature_names=None, class_names=None,
                 label="all", filled=False, impurity=True, node_ids=False,
                 proportion=False, rounded=False, precision=3, fontsize=None):
        super().__init__(max_depth=max_depth, feature_names=feature_names,
                         class_names=class_names, label=label, filled=filled,
                         impurity=impurity, node_ids=node_ids, proportion=proportion,
                         rounded=rounded, precision=precision)
        self.fontsize = fontsize
        self.ranks = {"leaves": []}
        self.colors = {"bounds": None}
        self.characters = ["#", "[", "]", "<=", "\n", "", ""]
        self.in_symbol = "in"
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args["boxstyle"] = "round"
        self.arrow_args = dict(arrowstyle="<-")

    def _make_tree(self, node_id, et, criterion, depth=0):
        name = self.node_to_str(et, node_id, criterion=criterion)
        if et.children_left[node_id] != TREE_LEAF and (self.max_depth is None or depth <= self.max_depth):
            children = [
                self._make_tree(et.children_left[node_id], et, criterion, depth=depth + 1),
                self._make_tree(et.children_right[node_id], et, criterion, depth=depth + 1),
            ]
        else:
            return _RTTree(name, node_id)
        return _RTTree(name, node_id, *children)

    def export(self, tree, criterion, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(0, tree, criterion)
        draw_tree = _buchheim(my_tree)

        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height
        scale_x = ax_width / max_x
        scale_y = ax_height / max_y
        self.recurse(draw_tree, tree, ax, max_x, max_y)

        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]
        renderer = ax.figure.canvas.get_renderer()
        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            extents = [
                bbox_patch.get_window_extent()
                for ann in anns
                if (bbox_patch := ann.get_bbox_patch()) is not None
            ]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            size = anns[0].get_fontsize() * min(scale_x / max_width, scale_y / max_height)
            for ann in anns:
                ann.set_fontsize(size)
        return anns

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        import matplotlib.pyplot as plt

        common_kwargs = dict(zorder=100 - 10 * depth, xycoords="axes fraction")
        if self.fontsize is not None:
            common_kwargs["fontsize"] = self.fontsize
        kwargs = dict(ha="center", va="center", bbox=self.bbox_args.copy(),
                      arrowprops=self.arrow_args.copy(), **common_kwargs)
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()

            if node.parent is None:
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = ((node.parent.x + 0.5) / max_x, (max_y - node.parent.y - 0.5) / max_y)
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
                if node.parent.parent is None:
                    text_pos = ((xy_parent[0] + xy[0]) / 2, (xy_parent[1] + xy[1]) / 2)
                    if node.parent.left() == node:
                        label_text, label_ha = ("True  ", "right")
                    else:
                        label_text, label_ha = ("  False", "left")
                    ax.annotate(label_text, text_pos, ha=label_ha, **common_kwargs)
            for child in node.children:
                self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)
        else:
            xy_parent = ((node.parent.x + 0.5) / max_x, (max_y - node.parent.y - 0.5) / max_y)
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)


# -----------------------------------------------------------------------------
# Public API (same signatures as sklearn.tree, plus ``tree_index``)
# -----------------------------------------------------------------------------
def _check_common(label, max_depth, precision):
    if label not in ("all", "root", "none"):
        raise ValueError("label must be one of {'all', 'root', 'none'}, got %r" % (label,))
    if max_depth is not None and (not isinstance(max_depth, Integral) or max_depth < 0):
        raise ValueError("max_depth must be a non-negative int or None")
    if precision is not None and (not isinstance(precision, Integral) or precision < 0):
        raise ValueError("precision must be a non-negative int")


def plot_tree(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rounded=False,
    precision=3,
    ax=None,
    fontsize=None,
    tree_index=0,
):
    """Plot a c50py tree with matplotlib, exactly like :func:`sklearn.tree.plot_tree`.

    Same parameters, defaults and look as scikit-learn.  The only extra
    parameter is ``tree_index``, which selects the tree to draw in a boosted
    model (``trials > 1``).

    Parameters
    ----------
    decision_tree : C5Classifier or C5Regressor
        A fitted c50py estimator.
    max_depth : int, default=None
        Maximum depth of the representation. If None, the tree is fully drawn.
    feature_names : array-like of str, default=None
        Names of the features. If None, the names seen in ``fit`` are used
        (a pandas DataFrame or the ``feature_names`` argument); if there are
        none, generic names ``x[0], x[1], ...``.
    class_names : array-like of str or True, default=None
        Names of the classes in the order of ``classes_``. ``True`` shows
        ``y[0], y[1], ...``. Ignored for regressors.
    label : {'all', 'root', 'none'}, default='all'
        Where to show the informative labels (``entropy = ``, ``samples = ``...).
    filled : bool, default=False
        Paint nodes by majority class (classification) or by the size of the
        prediction (regression), with scikit-learn's palette.
    impurity : bool, default=True
        Show the impurity of each node (entropy for classifiers, squared error
        for regressors).
    node_ids : bool, default=False
        Show the id of each node.
    proportion : bool, default=False
        Show ``value`` as proportions and ``samples`` as percentages.
    rounded : bool, default=False
        Draw boxes with rounded corners.
    precision : int, default=3
        Digits for impurity, thresholds and values.
    ax : matplotlib axis, default=None
        Axis to draw on. If None, the current axis is used (and cleared).
    fontsize : int, default=None
        Font size. If None, it is chosen to fit the figure.
    tree_index : int, default=0
        For boosted models, which of the trees to draw.

    Returns
    -------
    annotations : list of matplotlib artists
    """
    _check_common(label, max_depth, precision)
    ft = _flatten(decision_tree, tree_index)
    feature_names = _default_feature_names(decision_tree, feature_names)
    _check_feature_names(feature_names, ft)
    exporter = _MPLTreeExporter(
        max_depth=max_depth, feature_names=feature_names, class_names=class_names,
        label=label, filled=filled, impurity=impurity, node_ids=node_ids,
        proportion=proportion, rounded=rounded, precision=precision, fontsize=fontsize,
    )
    return exporter.export(ft, _criterion(decision_tree), ax=ax)


def export_graphviz(
    decision_tree,
    out_file=None,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    leaves_parallel=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rotate=False,
    rounded=False,
    special_characters=False,
    precision=3,
    fontname="helvetica",
    tree_index=0,
):
    """Export a c50py tree in DOT format, exactly like :func:`sklearn.tree.export_graphviz`.

    Does not need the ``graphviz`` package: the DOT text is written directly.
    If ``out_file`` is None the DOT source is returned as a string; otherwise
    it is written to that path or file handle.  Render it with
    ``graphviz.Source(dot)`` in a notebook, or ``dot -Tpng tree.dot -o tree.png``.
    """
    _check_common(label, max_depth, precision)
    ft = _flatten(decision_tree, tree_index)
    feature_names = _default_feature_names(decision_tree, feature_names)
    _check_feature_names(feature_names, ft)
    if feature_names is not None and any(not isinstance(n, str) for n in feature_names):
        raise ValueError("All feature names must be strings.")

    own_file = False
    return_string = False
    try:
        if isinstance(out_file, str):
            out_file = open(out_file, "w", encoding="utf-8")
            own_file = True
        if out_file is None:
            return_string = True
            out_file = StringIO()
        exporter = _DOTTreeExporter(
            out_file=out_file, max_depth=max_depth, feature_names=feature_names,
            class_names=class_names, label=label, filled=filled,
            leaves_parallel=leaves_parallel, impurity=impurity, node_ids=node_ids,
            proportion=proportion, rotate=rotate, rounded=rounded,
            special_characters=special_characters, precision=precision, fontname=fontname,
        )
        exporter.export(ft, _criterion(decision_tree))
        if return_string:
            return exporter.out_file.getvalue()
    finally:
        if own_file:
            out_file.close()


def _compute_depth(tree, node):
    def rec(n, d):
        l, r = tree.children_left[n], tree.children_right[n]
        if l != -1 and r != -1:
            return max(rec(l, d + 1), rec(r, d + 1))
        return d
    return rec(node, 1)


def export_text(
    decision_tree,
    *,
    feature_names=None,
    class_names=None,
    max_depth=10,
    spacing=3,
    decimals=2,
    show_weights=False,
    tree_index=0,
):
    """Text report of the tree rules, exactly like :func:`sklearn.tree.export_text`.

    Categorical splits are written ``feature in {a, b}`` / ``feature not in {a, b}``.
    """
    ft = _flatten(decision_tree, tree_index)
    clf = _is_classifier(decision_tree)
    feature_names = _default_feature_names(decision_tree, feature_names)
    if clf:
        if class_names is None:
            class_names = list(decision_tree.classes_)
        elif len(class_names) != len(decision_tree.classes_):
            raise ValueError(
                "When `class_names` is an array, it should contain as many items as "
                f"`decision_tree.classes_`. Got {len(class_names)} while the tree was "
                f"fitted with {len(decision_tree.classes_)} classes."
            )
    if feature_names is not None and ft.n_features and len(feature_names) != ft.n_features:
        raise ValueError("feature_names must contain %d elements, got %d" % (ft.n_features, len(feature_names)))

    def fname(i):
        return feature_names[i] if feature_names is not None else "feature_{}".format(i)

    value_fmt = ("{}{} weights: {}\n" if show_weights else "{}{}{}\n") if clf else "{}{} value: {}\n"
    report = StringIO()

    def add_leaf(value, wn, class_name, indent):
        if clf:
            val = ""
            if show_weights:
                val = "[" + ", ".join("{1:.{0}f}".format(decimals, v * wn) for v in value) + "]"
            val += " class: " + str(class_name)
        else:
            val = "[" + ", ".join("{1:.{0}f}".format(decimals, v) for v in value) + "]"
        report.write(value_fmt.format(indent, "", val))

    def rec(node, depth):
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing
        value = ft.value[node][0]
        class_name = class_names[int(np.argmax(value))] if clf else None
        wn = ft.weighted_n_node_samples[node]
        if depth <= max_depth + 1:
            if ft.children_left[node] != TREE_LEAF:
                name = fname(ft.feature[node])
                if ft.split_type[node] == "categorical":
                    cats = "{" + ", ".join(map(str, ft.categories[node])) + "}"
                    report.write("{} {} in {}\n".format(indent, name, cats))
                    rec(ft.children_left[node], depth + 1)
                    report.write("{} {} not in {}\n".format(indent, name, cats))
                    rec(ft.children_right[node], depth + 1)
                else:
                    thr = "{1:.{0}f}".format(decimals, ft.threshold[node])
                    report.write("{} {} <= {}\n".format(indent, name, thr))
                    rec(ft.children_left[node], depth + 1)
                    report.write("{} {} >  {}\n".format(indent, name, thr))
                    rec(ft.children_right[node], depth + 1)
            else:
                add_leaf(value, wn, class_name, indent)
        else:
            sd = _compute_depth(ft, node)
            if sd == 1:
                add_leaf(value, wn, class_name, indent)
            else:
                report.write("{} {}\n".format(indent, "truncated branch of depth %d" % sd))

    rec(0, 1)
    return report.getvalue()


# -----------------------------------------------------------------------------
# Estimator methods
# -----------------------------------------------------------------------------
class _TreeExportMixin:
    """Adds ``plot_tree``, ``export_graphviz`` and ``export_text`` methods.

    They are thin wrappers around the module-level functions of the same name,
    so ``model.plot_tree(...)`` and ``c50py.plot_tree(model, ...)`` are the same.
    """

    def plot_tree(self, *, max_depth=None, feature_names=None, class_names=None,
                  label="all", filled=False, impurity=True, node_ids=False,
                  proportion=False, rounded=False, precision=3, ax=None,
                  fontsize=None, tree_index=0):
        """Draw the tree with matplotlib, like :func:`sklearn.tree.plot_tree`.

        See :func:`c50py.plot_tree` for the parameters.
        """
        return plot_tree(self, max_depth=max_depth, feature_names=feature_names,
                         class_names=class_names, label=label, filled=filled,
                         impurity=impurity, node_ids=node_ids, proportion=proportion,
                         rounded=rounded, precision=precision, ax=ax, fontsize=fontsize,
                         tree_index=tree_index)

    def export_text(self, *, feature_names=None, class_names=None, max_depth=10,
                    spacing=3, decimals=2, show_weights=False, tree_index=0):
        """Text report of the rules, like :func:`sklearn.tree.export_text`."""
        return export_text(self, feature_names=feature_names, class_names=class_names,
                           max_depth=max_depth, spacing=spacing, decimals=decimals,
                           show_weights=show_weights, tree_index=tree_index)

    def export_graphviz(self, out_file=None, feature_names=None, *, class_names=None,
                        max_depth=None, label="all", filled=False, leaves_parallel=False,
                        impurity=True, node_ids=False, proportion=False, rotate=False,
                        rounded=False, special_characters=False, precision=3,
                        fontname="helvetica", tree_index=0, format=None, filename=None):
        """Export the tree in DOT format, like :func:`sklearn.tree.export_graphviz`.

        * ``out_file=None`` (default): returns the DOT source as a string.
        * ``out_file="tree.dot"`` or a file handle: writes the DOT source there.

        Backwards compatible with c50py <= 0.3: when ``format`` is given,
        ``out_file`` (or ``filename``) is a *basename*; ``format="dot"`` writes
        ``<basename>.dot`` and any other format (``"png"``, ``"pdf"``, ``"svg"``)
        is rendered with Graphviz to ``<basename>.<format>`` (falling back to a
        ``.dot`` file if the Graphviz program is missing).  The path is returned.
        With ``format`` and no file name the DOT source is returned.
        """
        if filename is not None:
            out_file = filename
        kw = dict(max_depth=max_depth, feature_names=feature_names, class_names=class_names,
                  label=label, filled=filled, leaves_parallel=leaves_parallel,
                  impurity=impurity, node_ids=node_ids, proportion=proportion,
                  rotate=rotate, rounded=rounded, special_characters=special_characters,
                  precision=precision, fontname=fontname, tree_index=tree_index)
        if format is None or not isinstance(out_file, str):
            return export_graphviz(self, out_file, **kw)
        # legacy behaviour: basename + format
        source = export_graphviz(self, None, **kw)
        if format.lower() in ("dot", "gv"):
            path = f"{out_file}.dot"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(source)
            return path
        try:
            import graphviz
            graphviz.Source(source, format=format).render(out_file, cleanup=True)
            return f"{out_file}.{format}"
        except Exception:
            path = f"{out_file}.dot"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(source)
            return path
