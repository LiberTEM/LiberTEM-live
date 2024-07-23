[Bugfix] Make the MIB simulator a bit more realistic
====================================================

* Allow multiple acquisitions per connection in triggered mode (:pr:`159`).
* Fix sequence number in continuous mode; fix generated acquisition header for
  the continuous case
* Fix control socket handling
