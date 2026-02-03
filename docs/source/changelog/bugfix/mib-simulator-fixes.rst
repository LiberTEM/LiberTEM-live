[Bugfix] Make the MIB simulator a bit more realistic
====================================================

* Allow multiple acquisitions per connection in triggered and untriggered mode (:pr:`159`, :pr:`191`).
* Fix sequence number in continuous mode; fix generated acquisition header for
  the continuous case
* Fix control socket handling
