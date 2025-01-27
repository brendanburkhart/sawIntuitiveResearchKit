# (C) Copyright 2025 Johns Hopkins University (JHU), All Rights Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---

cmake_minimum_required (VERSION 3.10)

# create a list of required cisst libraries
set (REQUIRED_CISST_LIBRARIES
  cisstCommon
  cisstNumerical
  cisstVector
  cisstRobot)

# find cisst and make sure the required libraries have been compiled
find_package (cisst 1.3.0 REQUIRED ${REQUIRED_CISST_LIBRARIES})

if (cisst_FOUND_AS_REQUIRED)

  # load cisst configuration
  include (${CISST_USE_FILE})

  add_executable (sawIntuitiveResearchKitMassTestData ArtificialData.cpp MassDetermination.hpp)
  set_target_properties (sawIntuitiveResearchKitMassTestData PROPERTIES
                          COMPONENT sawIntuitiveResearchKit-Applications
                          FOLDER "sawIntuitiveResearchKit")

  add_executable (sawIntuitiveResearchKitMassEquations ConstructMassEquations.cpp MassDetermination.hpp)
  set_target_properties (sawIntuitiveResearchKitMassEquations PROPERTIES
                          COMPONENT sawIntuitiveResearchKit-Applications
                          FOLDER "sawIntuitiveResearchKit")

  # link against cisst libraries (and dependencies)
  cisst_target_link_libraries (sawIntuitiveResearchKitMassTestData ${REQUIRED_CISST_LIBRARIES})
  cisst_target_link_libraries (sawIntuitiveResearchKitMassEquations ${REQUIRED_CISST_LIBRARIES})

  install (TARGETS sawIntuitiveResearchKitMassTestData
            COMPONENT sawIntuitiveResearchKit-Applications
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)

  install (TARGETS sawIntuitiveResearchKitMassEquations
            COMPONENT sawIntuitiveResearchKit-Applications
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)

endif (cisst_FOUND_AS_REQUIRED)
