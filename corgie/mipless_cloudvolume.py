import json
import six
import copy
import cachetools

import cloudvolume
from cloudvolume import CloudVolume, Storage

from corgie.log import logger as corgie_logger


def jsonize_key(*kargs, **kwargs):
    result = ""
    for k in kargs[1:]:
        result += json.dumps(k)
        result += "_"

    result += json.dumps(kwargs, sort_keys=True)
    return result


def cv_is_cached(*kargs, **kwargs):
    key = jsonize_key(*kargs, **kwargs)
    return key in cv_cache


cv_cache = cachetools.LRUCache(maxsize=500)


class CachedCloudVolume(CloudVolume):
    @cachetools.cached(cv_cache, key=jsonize_key)
    def __new__(self, *args, **kwargs):
        return super().__new__(self, *args, **kwargs)


def deserialize_miplessCV_old(s, cache={}):
    if s in cache:
        return cache[s]
    else:
        contents = json.loads(s)
        mcv = MiplessCloudVolume(
            contents["path"], mkdir=contents["mkdir"], **contents["kwargs"]
        )
        cache[s] = mcv
        return mcv


def deserialize_miplessCV_old2(s, cache={}):
    cv_kwargs = {
        "bounded": False,
        "progress": False,
        "autocrop": False,
        "non_aligned_writes": False,
        "cdn_cache": False,
    }
    if s in cache:
        return cache[s]
    else:
        contents = json.loads(s)
        mcv = MiplessCloudVolume(
            contents["path"], mkdir=False, fill_missing=True, **cv_kwargs
        )
        cache[s] = mcv
        return mcv


def deserialize_miplessCV(s, cache={}):
    cv_kwargs = {
        "bounded": False,
        "progress": False,
        "autocrop": False,
        "non_aligned_writes": False,
        "cdn_cache": False,
    }
    if s in cache:
        return cache[s]
    else:
        mcv = MiplessCloudVolume(s, mkdir=False, fill_missing=True, **cv_kwargs)
        cache[s] = mcv
        return mcv


class MiplessCloudVolume:
    """Multi-mip access to CloudVolumes using the same path"""

    def __init__(
        self,
        path,
        info=None,
        allow_info_writes=True,
        obj=CachedCloudVolume,
        default_chunk=(512, 512, 1),
        overwrite=False,
        **kwargs
    ):
        self.path = path
        self.allow_info_writes = allow_info_writes
        self.cv_params = {}
        self.cv_params["info"] = info
        if "cv_params" in kwargs:
            self.cv_params.update(kwargs["cv_params"])
        self.cv_params.setdefault("bounded", False)
        self.cv_params.setdefault("progress", False)
        self.cv_params.setdefault("autocrop", False)
        self.cv_params.setdefault("non_aligned_writes", False)
        self.cv_params.setdefault("cdn_cache", False)
        self.cv_params.setdefault("fill_missing", True)
        self.cv_params.setdefault("delete_black_uploads", True)
        self.cv_params.setdefault("agglomerate", True)

        # for k, v in six.iteritems(kwargs):
        #     self.cv_params[k] = v

        self.default_chunk = default_chunk

        self.obj = obj
        self.cvs = {}

        self.info = None
        self.fetch_info()

        if "info" in self.cv_params and overwrite:
            self.store_info()

    # def exists(self):
    #       s = Storage(self.path)
    #       return s.exists('info')

    def fetch_info(self):
        print("Fetching info")
        tmp_cv = self.obj(self.path, **self.cv_params)
        self.info = tmp_cv.info

    def serialize(self):
        contents = {
            "path": self.path,
            "allow_info_writes": self.allow_info_writes,
            "cv_params": self.cv_params,
        }
        s = json.dumps(contents)
        return s

    @classmethod
    def deserialize(cls, s, cache={}, **kwargs):
        if s in cache:
            return cache[s]
        else:
            import pdb

            pdb.set_trace()
            mcv = cls(s, **kwargs)
            cache[s] = mcv
            return mcv

    def get_info(self):
        return self.info

    def store_info(self, info=None):
        if not self.allow_info_writes:
            raise Exception(
                "Attempting to store info to {}, but "
                "'allow_info_writes' flag is set to False".format(self.path)
            )

        tmp_cv = self.obj(self.path, **self.cv_params)
        if info is not None:
            tmp_cv.info = info

        tmp_cv.commit_info()
        tmp_cv.commit_provenance()

    def ensure_info_has_mip(self, mip):
        tmp_cv = self.obj(self.path, **self.cv_params)
        scale_num = len(tmp_cv.info["scales"])

        if scale_num < mip + 1:
            while scale_num < mip + 1:
                tmp_cv.add_scale(
                    (2 ** scale_num, 2 ** scale_num, 1), chunk_size=self.default_chunk
                )
                scale_num += 1

            self.store_info(tmp_cv.info)

    def extend_info_to_mip(self, mip):
        info = self.get_info()
        highest_mip = len(info["scales"]) - 1
        highest_mip_info = info["scales"][-1]

        if highest_mip >= mip:
            return

        while highest_mip < mip:
            new_highest_mip_info = copy.deepcopy(highest_mip_info)
            # size
            # voxel offset
            # resolution -> key
            new_highest_mip_info["size"] = [
                highest_mip_info["size"][0] // 2,
                highest_mip_info["size"][1] // 2,
                highest_mip_info["size"][2],
            ]

            new_highest_mip_info["voxel_offset"] = [
                highest_mip_info["voxel_offset"][0] // 2,
                highest_mip_info["voxel_offset"][1] // 2,
                highest_mip_info["voxel_offset"][2],
            ]

            new_highest_mip_info["resolution"] = [
                highest_mip_info["resolution"][0] * 2,
                highest_mip_info["resolution"][1] * 2,
                highest_mip_info["resolution"][2],
            ]

            new_highest_mip_info["key"] = "_".join(
                [str(i) for i in new_highest_mip_info["resolution"]]
            )
            info["scales"].append(new_highest_mip_info)
            highest_mip += 1
            highest_mip_info = new_highest_mip_info

        self.store_info()

    def create(self, mip):
        corgie_logger.debug(
            "Creating CloudVolume for {0} at MIP{1}".format(self.path, mip)
        )
        self.extend_info_to_mip(mip)

        self.cvs[mip] = self.obj(self.path, mip=mip, **self.cv_params)

        # if self.mkdir:
        #  self.cvs[mip].commit_info()
        #  self.cvs[mip].commit_provenance()

    def __getitem__(self, mip):
        if mip not in self.cvs:
            self.create(mip)
        return self.cvs[mip]

    def __repr__(self):
        return self.path
