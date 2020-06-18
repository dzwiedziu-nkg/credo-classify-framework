.. toctree::
   :numbered:
   :maxdepth: 2
   :caption: Contents:

.. mdinclude:: ../README.md

API Documentation
-----------------

The IO functions
^^^^^^^^^^^^^^^^

.. automodule:: credo_cf.io.load_write
   :members: load_json_from_stream, load_json, serialize, deserialize

Helpers functions for IO
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: credo_cf.io.io_utils
   :members: progress_and_process_image, progress_load_filter

Grouping functions
^^^^^^^^^^^^^^^^^^

.. automodule:: credo_cf.commons.grouping
   :members: group_by_lambda, get_and_set, group_by_device_id, group_by_resolution, group_by_timestamp_division, get_resolution_key

Classification functions
^^^^^^^^^^^^^^^^^^^^^^^^

Working on list of detections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: credo_cf.classification.artifact
   :members: too_often, hot_pixel, near_hot_pixel, near_hot_pixel2

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
