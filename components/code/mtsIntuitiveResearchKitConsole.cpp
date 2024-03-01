/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  Author(s):  Anton Deguet
  Created on: 2013-05-17

  (C) Copyright 2013-2024 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

// system include
#include <iostream>

// cisst
#include <cisstCommon/cmnClassRegister.h>
#include <cisstCommon/cmnRandomSequence.h>
#include <cisstOSAbstraction/osaDynamicLoader.h>

#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <cisstMultiTask/mtsManagerLocal.h>
#include <cisstParameterTypes/prmEventButton.h>

#include <sawTextToSpeech/mtsTextToSpeech.h>
#include <sawRobotIO1394/mtsRobotIO1394.h>
#include <sawControllers/mtsPID.h>

#include <sawIntuitiveResearchKit/sawIntuitiveResearchKitRevision.h>
#include <sawIntuitiveResearchKit/sawIntuitiveResearchKitConfig.h>
#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKitMTM.h>
#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKitPSM.h>
#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKitECM.h>
#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKitSUJ.h>
#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKitSUJSi.h>
#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKitSUJFixed.h>
#include <sawIntuitiveResearchKit/mtsSocketClientPSM.h>
#include <sawIntuitiveResearchKit/mtsSocketServerPSM.h>
#include <sawIntuitiveResearchKit/mtsDaVinciHeadSensor.h>
#if sawIntuitiveResearchKit_HAS_HID_HEAD_SENSOR
#include <sawIntuitiveResearchKit/mtsHIDHeadSensor.h>
#endif
#include <sawIntuitiveResearchKit/mtsDaVinciEndoscopeFocus.h>
#include <sawIntuitiveResearchKit/mtsTeleOperationPSM.h>
#include <sawIntuitiveResearchKit/mtsTeleOperationECM.h>
#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKitConsole.h>

#include <json/json.h>

CMN_IMPLEMENT_SERVICES(mtsIntuitiveResearchKitConsole);

bool mtsIntuitiveResearchKitConsole::Arm::native_or_derived(void) const
{
    switch (m_type) {
    case ARM_MTM:
    case ARM_PSM:
    case ARM_ECM:
    case ARM_MTM_DERIVED:
    case ARM_PSM_DERIVED:
    case ARM_ECM_DERIVED:
    case ARM_SUJ_Classic:
    case ARM_SUJ_Si:
    case ARM_SUJ_Fixed:
    case FOCUS_CONTROLLER:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

bool mtsIntuitiveResearchKitConsole::Arm::psm(void) const
{
    switch (m_type) {
    case ARM_PSM:
    case ARM_PSM_DERIVED:
    case ARM_PSM_GENERIC:
    case ARM_PSM_SOCKET:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

bool mtsIntuitiveResearchKitConsole::Arm::mtm(void) const
{
    switch (m_type) {
    case ARM_MTM:
    case ARM_MTM_DERIVED:
    case ARM_MTM_GENERIC:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

bool mtsIntuitiveResearchKitConsole::Arm::ecm(void) const
{
    switch (m_type) {
    case ARM_ECM:
    case ARM_ECM_DERIVED:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

bool mtsIntuitiveResearchKitConsole::Arm::suj(void) const
{
    switch (m_type) {
    case ARM_SUJ_Classic:
    case ARM_SUJ_Si:
    case ARM_SUJ_Fixed :
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

bool mtsIntuitiveResearchKitConsole::Arm::generic(void) const
{
    switch (m_type) {
    case ARM_MTM_GENERIC:
    case ARM_PSM_GENERIC:
    case ARM_ECM_GENERIC:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

bool mtsIntuitiveResearchKitConsole::Arm::native_or_derived_mtm(void) const
{
    switch (m_type) {
    case ARM_MTM:
    case ARM_MTM_DERIVED:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

bool mtsIntuitiveResearchKitConsole::Arm::native_or_derived_psm(void) const
{
    switch (m_type) {
    case ARM_PSM:
    case ARM_PSM_DERIVED:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

bool mtsIntuitiveResearchKitConsole::Arm::native_or_derived_ecm(void) const
{
    switch (m_type) {
    case ARM_ECM:
    case ARM_ECM_DERIVED:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

mtsIntuitiveResearchKitArm::GenerationType mtsIntuitiveResearchKitConsole::Arm::generation(void) const
{
    if (m_generation == mtsIntuitiveResearchKitArm::GENERATION_UNDEFINED) {
        CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::generation: trying to access generation before it is set"
                           << std::endl;
        exit(EXIT_FAILURE);
    }
    return m_generation;
}

bool mtsIntuitiveResearchKitConsole::Arm::expects_PID(void) const
{
    return (native_or_derived()
            && !suj());
}

bool mtsIntuitiveResearchKitConsole::Arm::expects_IO(void) const
{
    return (native_or_derived()
            && (m_type != Arm::ARM_PSM_SOCKET)
            && (m_type != Arm::ARM_SUJ_Si)
            && (m_type != Arm::ARM_SUJ_Fixed)
            && (m_simulation == Arm::SIMULATION_NONE));
}

mtsIntuitiveResearchKitConsole::Arm::Arm(mtsIntuitiveResearchKitConsole * console,
                                         const std::string & name,
                                         const std::string & ioComponentName):
    m_console(console),
    m_name(name),
    m_IO_component_name(ioComponentName),
    m_arm_period(mtsIntuitiveResearchKit::ArmPeriod),
    mSUJClutched(false)
{}

void mtsIntuitiveResearchKitConsole::Arm::ConfigurePID(const std::string & pid_config_,
                                                       const double & periodInSeconds)
{
    std::string pid_config = pid_config_;
    // not user defined, try to find the default
    if (pid_config == "") {
        if (native_or_derived_mtm()) {
            pid_config = "pid/sawControllersPID-MTM.json";
        } else if (native_or_derived_psm()) {
            if (generation() == mtsIntuitiveResearchKitArm::GENERATION_Classic) {
                pid_config = "pid/sawControllersPID-PSM.json";
            } else {
                pid_config = "pid/sawControllersPID-PSM-Si.json";
            }
        } else if (native_or_derived_ecm()) {
            if (generation() == mtsIntuitiveResearchKitArm::GENERATION_Classic) {
                pid_config = "pid/sawControllersPID-ECM.json";
            } else  {
                pid_config = "pid/sawControllersPID-ECM-Si.json";
            }
        } else {
            pid_config = "pid/sawControllersPID-" + m_name + ".json";
        }
        CMN_LOG_INIT_VERBOSE << "ConfigurePID: can't find \"pid\" setting for arm \""
                             << m_name << "\", using default: \""
                             << pid_config << "\"" << std::endl;
    }

    m_PID_configuration_file = m_config_path.Find(pid_config);
    if (m_PID_configuration_file == "") {
        CMN_LOG_INIT_ERROR << "ConfigurePID: can't find PID file " << pid_config << std::endl;
        exit(EXIT_FAILURE);
    }

    m_PID_component_name = m_name + "-PID";

    mtsManagerLocal * componentManager = mtsManagerLocal::GetInstance();
    mtsPID * pid = new mtsPID(m_PID_component_name,
                              (periodInSeconds != 0.0) ? periodInSeconds : mtsIntuitiveResearchKit::IOPeriod);
    bool hasIO = true;
    pid->Configure(m_PID_configuration_file);
    if (m_simulation == SIMULATION_KINEMATIC) {
        pid->SetSimulated();
        hasIO = false;
    }
    componentManager->AddComponent(pid);
    if (hasIO) {
        m_console->mConnections.Add(PIDComponentName(), "RobotJointTorqueInterface",
                                    IOComponentName(), Name());
        if (periodInSeconds == 0.0) {
            m_console->mConnections.Add(PIDComponentName(), "ExecIn",
                                        IOComponentName(), "ExecOut");
        }
    }
}

void mtsIntuitiveResearchKitConsole::Arm::ConfigureArm(const ArmType arm_type,
                                                       const std::string & kinematicsConfigFile,
                                                       const double & periodInSeconds)
{
    m_type = arm_type;
    bool armPSMOrDerived = false;
    bool armECMOrDerived = false;

    mtsManagerLocal * componentManager = mtsManagerLocal::GetInstance();
    m_arm_configuration_file = kinematicsConfigFile;
    // for research kit arms, create, add to manager and connect to
    // extra IO, PID, etc.  For generic arms, do nothing.
    switch (arm_type) {
    case ARM_MTM:
        {
            mtsIntuitiveResearchKitMTM * mtm = new mtsIntuitiveResearchKitMTM(Name(), periodInSeconds);
            if (m_simulation == SIMULATION_KINEMATIC) {
                mtm->set_simulated();
            }
            mtm->set_calibration_mode(m_calibration_mode);
            mtm->Configure(m_arm_configuration_file);
            m_generation = mtm->generation();
            SetBaseFrameIfNeeded(mtm);
            componentManager->AddComponent(mtm);
        }
        break;
    case ARM_PSM:
        armPSMOrDerived = true;
        {
            mtsIntuitiveResearchKitPSM * psm = new mtsIntuitiveResearchKitPSM(Name(), periodInSeconds);
            if (m_simulation == SIMULATION_KINEMATIC) {
                psm->set_simulated();
            }
            psm->set_calibration_mode(m_calibration_mode);
            psm->Configure(m_arm_configuration_file);
            m_generation = psm->generation();
            SetBaseFrameIfNeeded(psm);
            componentManager->AddComponent(psm);

            if (m_socket_server) {
                mtsSocketServerPSM * serverPSM = new mtsSocketServerPSM(SocketComponentName(), periodInSeconds, m_IP, m_port);
                serverPSM->Configure();
                componentManager->AddComponent(serverPSM);
                m_console->mConnections.Add(SocketComponentName(), "PSM",
                                            ComponentName(), InterfaceName());
            }
        }
        break;
    case ARM_PSM_SOCKET:
        {
            mtsSocketClientPSM * clientPSM = new mtsSocketClientPSM(Name(), periodInSeconds, m_IP, m_port);
            clientPSM->Configure();
            componentManager->AddComponent(clientPSM);
        }
        break;
    case ARM_ECM:
        armECMOrDerived = true;
        {
            mtsIntuitiveResearchKitECM * ecm = new mtsIntuitiveResearchKitECM(Name(), periodInSeconds);
            if (m_simulation == SIMULATION_KINEMATIC) {
                ecm->set_simulated();
            }
            ecm->set_calibration_mode(m_calibration_mode);
            ecm->Configure(m_arm_configuration_file);
            m_generation = ecm->generation();
            SetBaseFrameIfNeeded(ecm);
            componentManager->AddComponent(ecm);
        }
        break;
    case ARM_SUJ_Classic:
        {
            mtsIntuitiveResearchKitSUJ * suj = new mtsIntuitiveResearchKitSUJ(Name(), periodInSeconds);
            if (m_simulation == SIMULATION_KINEMATIC) {
                suj->set_simulated();
            } else if (m_simulation == SIMULATION_NONE) {
                m_console->mConnections.Add(Name(), "NoMuxReset",
                                            IOComponentName(), "NoMuxReset");
                m_console->mConnections.Add(Name(), "MuxIncrement",
                                            IOComponentName(), "MuxIncrement");
                m_console->mConnections.Add(Name(), "ControlPWM",
                                            IOComponentName(), "ControlPWM");
                m_console->mConnections.Add(Name(), "DisablePWM",
                                            IOComponentName(), "DisablePWM");
                m_console->mConnections.Add(Name(), "MotorUp",
                                            IOComponentName(), "MotorUp");
                m_console->mConnections.Add(Name(), "MotorDown",
                                            IOComponentName(), "MotorDown");
                m_console->mConnections.Add(Name(), "SUJ-Clutch-1",
                                            IOComponentName(), "SUJ-Clutch-1");
                m_console->mConnections.Add(Name(), "SUJ-Clutch-2",
                                            IOComponentName(), "SUJ-Clutch-2");
                m_console->mConnections.Add(Name(), "SUJ-Clutch-3",
                                            IOComponentName(), "SUJ-Clutch-3");
                m_console->mConnections.Add(Name(), "SUJ-Clutch-4",
                                            IOComponentName(), "SUJ-Clutch-4");
            }
            suj->Configure(m_arm_configuration_file);
            componentManager->AddComponent(suj);
        }
        break;
    case ARM_SUJ_Si:
        {
#if sawIntuitiveResearchKit_HAS_SUJ_Si
            mtsIntuitiveResearchKitSUJSi * suj = new mtsIntuitiveResearchKitSUJSi(Name(), periodInSeconds);
            if (m_simulation == SIMULATION_KINEMATIC) {
                suj->set_simulated();
            }
            suj->Configure(m_arm_configuration_file);
            componentManager->AddComponent(suj);
#else
            CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureArm: can't create an arm of type SUJ_Si because sawIntuitiveResearchKit_HAS_SUJ_Si is set to OFF in CMake"
                               << std::endl;
            exit(EXIT_FAILURE);
#endif
        }
        break;
    case ARM_SUJ_Fixed:
        {
            mtsIntuitiveResearchKitSUJFixed * suj = new mtsIntuitiveResearchKitSUJFixed(Name(), periodInSeconds);
            suj->Configure(m_arm_configuration_file);
            componentManager->AddComponent(suj);
        }
        break;
    case ARM_MTM_DERIVED:
        {
            mtsComponent * component;
            component = componentManager->GetComponent(Name());
            if (component) {
                mtsIntuitiveResearchKitMTM * mtm = dynamic_cast<mtsIntuitiveResearchKitMTM *>(component);
                if (mtm) {
                    if (m_simulation == SIMULATION_KINEMATIC) {
                        mtm->set_simulated();
                    }
                    mtm->set_calibration_mode(m_calibration_mode);
                    mtm->Configure(m_arm_configuration_file);
                    m_generation = mtm->generation();
                    SetBaseFrameIfNeeded(mtm);
                } else {
                    CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureArm: component \""
                                       << Name() << "\" doesn't seem to be derived from mtsIntuitiveResearchKitMTM."
                                       << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureArm: component \""
                                   << Name() << "\" not found."
                                   << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        break;
    case ARM_PSM_DERIVED:
        armPSMOrDerived = true;
        {
            mtsComponent * component;
            component = componentManager->GetComponent(Name());
            if (component) {
                mtsIntuitiveResearchKitPSM * psm = dynamic_cast<mtsIntuitiveResearchKitPSM *>(component);
                if (psm) {
                    if (m_simulation == SIMULATION_KINEMATIC) {
                        psm->set_simulated();
                    }
                    psm->set_calibration_mode(m_calibration_mode);
                    psm->Configure(m_arm_configuration_file);
                    m_generation = psm->generation();
                    SetBaseFrameIfNeeded(psm);
                } else {
                    CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureArm: component \""
                                       << Name() << "\" doesn't seem to be derived from mtsIntuitiveResearchKitPSM."
                                       << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureArm: component \""
                                   << Name() << "\" not found."
                                   << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        break;
    case ARM_ECM_DERIVED:
        armECMOrDerived = true;
        {
            mtsComponent * component;
            component = componentManager->GetComponent(Name());
            if (component) {
                mtsIntuitiveResearchKitECM * ecm = dynamic_cast<mtsIntuitiveResearchKitECM *>(component);
                if (ecm) {
                    if (m_simulation == SIMULATION_KINEMATIC) {
                        ecm->set_simulated();
                    }
                    ecm->set_calibration_mode(m_calibration_mode);
                    ecm->Configure(m_arm_configuration_file);
                    m_generation = ecm->generation();
                    SetBaseFrameIfNeeded(ecm);
                } else {
                    CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureArm: component \""
                                       << Name() << "\" doesn't seem to be derived from mtsIntuitiveResearchKitECM."
                                       << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureArm: component \""
                                   << Name() << "\" not found."
                                   << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        break;

    default:
        break;
    }

    if (armPSMOrDerived && (m_simulation == SIMULATION_NONE)) {
        std::vector<std::string> itfs = {"Adapter", "Tool", "ManipClutch", "Dallas"};
        for (const auto & itf : itfs) {
            m_console->mConnections.Add(Name(), itf,
                                        IOComponentName(), Name() + "-" + itf);
        }
    }

    if (armECMOrDerived && (m_simulation == SIMULATION_NONE)) {
        m_console->mConnections.Add(Name(), "ManipClutch",
                                    IOComponentName(), Name() + "-ManipClutch");
    }

    // for Si patient side, connect the SUJ brakes to buttons on arm
    if ((armPSMOrDerived || armECMOrDerived)
        && (m_simulation == SIMULATION_NONE)
        && (generation() == mtsIntuitiveResearchKitArm::GENERATION_Si)) {
        std::vector<std::string> itfs = {"SUJClutch", "SUJClutch2", "SUJBrake"};
        for (const auto & itf : itfs) {
            m_console->mConnections.Add(Name(), itf,
                                        IOComponentName(), Name() + "-" + itf);
        }
    }
}

void mtsIntuitiveResearchKitConsole::Arm::SetBaseFrameIfNeeded(mtsIntuitiveResearchKitArm * arm_pointer)
{
    if (m_base_frame.ReferenceFrame() != "") {
        arm_pointer->set_base_frame(m_base_frame);
    }
}

bool mtsIntuitiveResearchKitConsole::Arm::Connect(void)
{
    mtsManagerLocal * componentManager = mtsManagerLocal::GetInstance();
    // if the arm is a research kit arm
    if (native_or_derived()) {
        // Connect arm to IO if not simulated
        if (expects_IO()) {
            componentManager->Connect(Name(), "RobotIO",
                                      IOComponentName(), Name());
        }
        // connect MTM gripper to IO
        if (((m_type == ARM_MTM)
             || (m_type == ARM_MTM_DERIVED))
            && (m_simulation == SIMULATION_NONE)) {
            componentManager->Connect(Name(), "GripperIO",
                                      IOComponentName(), Name() + "-Gripper");
        }
        // connect PID
        if (expects_PID()) {
            componentManager->Connect(Name(), "PID",
                                      PIDComponentName(), "Controller");
        }
        // connect m_base_frame if needed
        if ((m_base_frame_component_name != "") && (m_base_frame_interface_name != "")) {
            componentManager->Connect(m_base_frame_component_name, m_base_frame_interface_name,
                                      Name(), "Arm");
        }
    }
    return true;
}

const std::string & mtsIntuitiveResearchKitConsole::Arm::Name(void) const {
    return m_name;
}

const std::string & mtsIntuitiveResearchKitConsole::Arm::ComponentName(void) const {
    return m_arm_component_name;
}

const std::string & mtsIntuitiveResearchKitConsole::Arm::InterfaceName(void) const {
    return m_arm_interface_name;
}

const std::string & mtsIntuitiveResearchKitConsole::Arm::SocketComponentName(void) const {
    return m_socket_component_name;
}

const std::string & mtsIntuitiveResearchKitConsole::Arm::IOComponentName(void) const {
    return m_IO_component_name;
}

const std::string & mtsIntuitiveResearchKitConsole::Arm::PIDComponentName(void) const {
    return m_PID_component_name;
}

void mtsIntuitiveResearchKitConsole::Arm::CurrentStateEventHandler(const prmOperatingState & currentState)
{
    m_console->SetArmCurrentState(m_name, currentState);
}

mtsIntuitiveResearchKitConsole::TeleopECM::TeleopECM(const std::string & name):
    m_name(name)
{
}

void mtsIntuitiveResearchKitConsole::TeleopECM::ConfigureTeleop(const TeleopECMType type,
                                                                const double & periodInSeconds,
                                                                const Json::Value & jsonConfig)
{
    m_type = type;
    mtsManagerLocal * componentManager = mtsManagerLocal::GetInstance();

    switch (type) {
    case TELEOP_ECM:
        {
            mtsTeleOperationECM * teleop = new mtsTeleOperationECM(m_name, periodInSeconds);
            teleop->Configure(jsonConfig);
            componentManager->AddComponent(teleop);
        }
        break;
    case TELEOP_ECM_DERIVED:
        {
            mtsComponent * component;
            component = componentManager->GetComponent(Name());
            if (component) {
                mtsTeleOperationECM * teleop = dynamic_cast<mtsTeleOperationECM *>(component);
                if (teleop) {
                    teleop->Configure(jsonConfig);
                } else {
                    CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureTeleop: component \""
                                       << Name() << "\" doesn't seem to be derived from mtsTeleOperationECM."
                                       << std::endl;
                }
            } else {
                CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureTeleop: component \""
                                   << Name() << "\" not found."
                                   << std::endl;
            }
        }
        break;
    default:
        break;
    }
}

const std::string & mtsIntuitiveResearchKitConsole::TeleopECM::Name(void) const {
    return m_name;
}


mtsIntuitiveResearchKitConsole::TeleopPSM::TeleopPSM(const std::string & name,
                                                     const std::string & nameMTM,
                                                     const std::string & namePSM):
    mSelected(false),
    m_name(name),
    mMTMName(nameMTM),
    mPSMName(namePSM)
{
}

void mtsIntuitiveResearchKitConsole::TeleopPSM::ConfigureTeleop(const TeleopPSMType type,
                                                                const double & periodInSeconds,
                                                                const Json::Value & jsonConfig)
{
    m_type = type;
    mtsManagerLocal * componentManager = mtsManagerLocal::GetInstance();

    switch (type) {
    case TELEOP_PSM:
        {
            mtsTeleOperationPSM * teleop = new mtsTeleOperationPSM(m_name, periodInSeconds);
            teleop->Configure(jsonConfig);
            componentManager->AddComponent(teleop);
        }
        break;
    case TELEOP_PSM_DERIVED:
        {
            mtsComponent * component;
            component = componentManager->GetComponent(Name());
            if (component) {
                mtsTeleOperationPSM * teleop = dynamic_cast<mtsTeleOperationPSM *>(component);
                if (teleop) {
                    teleop->Configure(jsonConfig);
                } else {
                    CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureTeleop: component \""
                                       << Name() << "\" doesn't seem to be derived from mtsTeleOperationPSM."
                                       << std::endl;
                }
            } else {
                CMN_LOG_INIT_ERROR << "mtsIntuitiveResearchKitConsole::Arm::ConfigureTeleop: component \""
                                   << Name() << "\" not found."
                                   << std::endl;
            }
        }
        break;
    default:
        break;
    }
}

const std::string & mtsIntuitiveResearchKitConsole::TeleopPSM::Name(void) const {
    return m_name;
}



mtsIntuitiveResearchKitConsole::mtsIntuitiveResearchKitConsole(const std::string & componentName):
    mtsTaskFromSignal(componentName, 100),
    m_configured(false),
    mTimeOfLastErrorBeep(0.0),
    mTeleopMTMToCycle(""),
    mTeleopECM(0),
    mOperatorPresent(false),
    mCameraPressed(false),
    m_IO_component_name("io")
{
    // configure search path
    m_config_path.Add(cmnPath::GetWorkingDirectory());
    // add path to source/share directory to find common files.  This
    // will work as long as this component is located in the same
    // parent directory as the "shared" directory.
    m_config_path.Add(std::string(sawIntuitiveResearchKit_SOURCE_DIR) + "/../share", cmnPath::TAIL);
    // default installation directory
    m_config_path.Add(mtsIntuitiveResearchKit::DefaultInstallationDirectory, cmnPath::TAIL);

    mInterface = AddInterfaceProvided("Main");
    if (mInterface) {
        mInterface->AddMessageEvents();
        mInterface->AddCommandVoid(&mtsIntuitiveResearchKitConsole::power_off, this,
                                   "power_off");
        mInterface->AddCommandVoid(&mtsIntuitiveResearchKitConsole::power_on, this,
                                   "power_on");
        mInterface->AddCommandVoid(&mtsIntuitiveResearchKitConsole::home, this,
                                   "home");
        mInterface->AddEventWrite(ConfigurationEvents.ArmCurrentState,
                                  "ArmCurrentState", prmKeyValue());
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::teleop_enable, this,
                                    "teleop_enable", false);
        mInterface->AddEventWrite(console_events.teleop_enabled,
                                  "teleop_enabled", false);
        // manage tele-op
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::cycle_teleop_psm_by_mtm, this,
                                    "cycle_teleop_psm_by_mtm", std::string(""));
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::select_teleop_psm, this,
                                    "select_teleop_psm", prmKeyValue("mtm", "psm"));
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::set_scale, this,
                                    "set_scale", mtsIntuitiveResearchKit::TeleOperationPSM::Scale);
        mInterface->AddEventWrite(ConfigurationEvents.scale,
                                  "scale", mtsIntuitiveResearchKit::TeleOperationPSM::Scale);
        mInterface->AddEventWrite(ConfigurationEvents.teleop_psm_selected,
                                  "teleop_psm_selected", prmKeyValue("MTM", "PSM"));
        mInterface->AddEventWrite(ConfigurationEvents.teleop_psm_unselected,
                                  "teleop_psm_unselected", prmKeyValue("MTM", "PSM"));
        // audio
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::set_volume, this,
                                    "set_volume", m_audio_volume);
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::beep, this,
                                    "beep", vctDoubleVec());
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::string_to_speech, this,
                                    "string_to_speech", std::string());
        mInterface->AddEventWrite(audio.volume,
                                  "volume", m_audio_volume);
        // emulate foot pedal events
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::OperatorPresentEventHandler, this,
                                    "emulate_operator_present", prmEventButton());
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::ClutchEventHandler, this,
                                    "emulate_clutch", prmEventButton());
        mInterface->AddCommandWrite(&mtsIntuitiveResearchKitConsole::CameraEventHandler, this,
                                    "emulate_camera", prmEventButton());
        // misc.
        mInterface->AddCommandRead(&mtsIntuitiveResearchKitConsole::calibration_mode, this,
                                   "calibration_mode", false);
    }
}

void mtsIntuitiveResearchKitConsole::set_calibration_mode(const bool mode)
{
    m_calibration_mode = mode;
    if (m_calibration_mode) {
        std::stringstream message;
        message << "set_calibration_mode:" << std::endl
                << "----------------------------------------------------" << std::endl
                << " Warning:" << std::endl
                << " You're running the dVRK console in calibration mode." << std::endl
                << " You should only do this if you are currently calibrating" << std::endl
                << " potentiometers." << std::endl
                << "----------------------------------------------------";
        std::cerr << "mtsIntuitiveResearchKitConsole::" << message.str() << std::endl;
        CMN_LOG_CLASS_INIT_WARNING << message.str() << std::endl;
    }
}

const bool & mtsIntuitiveResearchKitConsole::calibration_mode(void) const
{
    return m_calibration_mode;
}

void mtsIntuitiveResearchKitConsole::calibration_mode(bool & result) const
{
    result = m_calibration_mode;
}

void mtsIntuitiveResearchKitConsole::Configure(const std::string & filename)
{
    m_configured = false;

    std::ifstream jsonStream;
    jsonStream.open(filename.c_str());

    Json::Value jsonConfig, jsonValue;
    Json::Reader jsonReader;
    if (!jsonReader.parse(jsonStream, jsonConfig)) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to parse configuration" << std::endl
                                 << "File: " << filename << std::endl << "Error(s):" << std::endl
                                 << jsonReader.getFormattedErrorMessages();
        this->m_configured = false;
        exit(EXIT_FAILURE);
    }

    CMN_LOG_CLASS_INIT_VERBOSE << "Configure: " << this->GetName()
                               << " using file \"" << filename << "\"" << std::endl
                               << "----> content of configuration file: " << std::endl
                               << jsonConfig << std::endl
                               << "<----" << std::endl;

    // base component configuration
    mtsComponent::ConfigureJSON(jsonConfig);

    // extract path of main json config file to search other files relative to it
    std::string fullname = m_config_path.Find(filename);
    std::string configDir = fullname.substr(0, fullname.find_last_of('/'));
    m_config_path.Add(configDir, cmnPath::TAIL);

    mtsComponentManager * manager = mtsComponentManager::GetInstance();

    // first, create all custom components and connections, i.e. dynamic loading and creation
    const Json::Value componentManager = jsonConfig["component-manager"];
    if (!componentManager.empty()) {
        if (!manager->ConfigureJSON(componentManager, m_config_path)) {
            CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to configure component-manager" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // add text to speech component for the whole system
    mTextToSpeech = new mtsTextToSpeech();
    manager->AddComponent(mTextToSpeech);
    mtsInterfaceRequired * textToSpeechInterface = this->AddInterfaceRequired("TextToSpeech");
    textToSpeechInterface->AddFunction("Beep", audio.beep);
    textToSpeechInterface->AddFunction("StringToSpeech", audio.string_to_speech);
    m_audio_volume = 0.5;

    // IO default settings
    double periodIO = mtsIntuitiveResearchKit::IOPeriod;
    std::string port = mtsRobotIO1394::DefaultPort();
    std::string protocol = mtsIntuitiveResearchKit::FireWireProtocol;
    double watchdogTimeout = mtsIntuitiveResearchKit::WatchdogTimeout;

    jsonValue = jsonConfig["chatty"];
    if (!jsonValue.empty()) {
        mChatty = jsonValue.asBool();
    } else {
        mChatty = false;
    }

    // get user preferences
    jsonValue = jsonConfig["io"];
    if (!jsonValue.empty()) {
        jsonValue = jsonConfig["io"]["firewire-protocol"];
        if (!jsonValue.empty()) {
            protocol = jsonValue.asString();
        }

        jsonValue = jsonConfig["io"]["period"];
        if (!jsonValue.empty()) {
            periodIO = jsonValue.asDouble();
            if (periodIO > 1.0 * cmn_ms) {
                std::stringstream message;
                message << "Configure:" << std::endl
                        << "----------------------------------------------------" << std::endl
                        << " Warning:" << std::endl
                        << "   The period provided is quite high, i.e. " << periodIO << std::endl
                        << "   seconds.  We strongly recommend you change it to" << std::endl
                        << "   a value below 1 ms, i.e. 0.001." << std::endl
                        << "----------------------------------------------------";
                std::cerr << "mtsIntuitiveResearchKitConsole::" << message.str() << std::endl;
                CMN_LOG_CLASS_INIT_WARNING << message.str() << std::endl;
            }
        }
        jsonValue = jsonConfig["io"]["port"];
        if (!jsonValue.empty()) {
            port = jsonValue.asString();
        }

        jsonValue = jsonConfig["io"]["watchdog-timeout"];
        if (!jsonValue.empty()) {
            watchdogTimeout = jsonValue.asDouble();
            if (watchdogTimeout > 300.0 * cmn_ms) {
                watchdogTimeout = 300.0 * cmn_ms;
                CMN_LOG_CLASS_INIT_WARNING << "Configure: io:watchdog-timeout has to be lower than 300 ms, it has been capped at 300 ms" << std::endl;
            }
            if (watchdogTimeout <= 0.0) {
                watchdogTimeout = 0.0;
                std::stringstream message;
                message << "Configure:" << std::endl
                        << "----------------------------------------------------" << std::endl
                        << " Warning:" << std::endl
                        << "   Setting the watchdog timeout to zero disables the" << std::endl
                        << "   watchdog.   We strongly recommend to no specify" << std::endl
                        << "   io:watchdog-timeout or set it around 10 ms." << std::endl
                        << "----------------------------------------------------";
                std::cerr << "mtsIntuitiveResearchKitConsole::" << message.str() << std::endl;
                CMN_LOG_CLASS_INIT_WARNING << message.str() << std::endl;
            }
        }
    } else {
        CMN_LOG_CLASS_INIT_VERBOSE << "Configure: using default io:period, io:port, io:firewire-protocol and io:watchdog-timeout" << std::endl;
    }
    CMN_LOG_CLASS_INIT_VERBOSE << "Configure:" << std::endl
                               << "     - Period IO is " << periodIO << std::endl
                               << "     - Port is " << port << std::endl
                               << "     - Protocol is " << protocol << std::endl
                               << "     - Watchdog timeout is " << watchdogTimeout << std::endl;

    const Json::Value arms = jsonConfig["arms"];
    for (unsigned int index = 0; index < arms.size(); ++index) {
        if (!ConfigureArmJSON(arms[index], m_IO_component_name)) {
            CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to configure arms[" << index << "]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // loop over all arms to check if IO is needed, also check if some IO configuration files are listed in "io"
    mHasIO = false;
    const auto end = mArms.end();
    for (auto iter = mArms.begin(); iter != end; ++iter) {
        std::string ioConfig = iter->second->m_IO_configuration_file;
        if (!ioConfig.empty()) {
            mHasIO = true;
        }
    }
    bool physicalFootpedalsRequired = true;
    jsonValue = jsonConfig["io"];
    if (!jsonValue.empty()) {
        // generic files
        Json::Value configFiles = jsonValue["configuration-files"];
        if (!configFiles.empty()) {
            mHasIO = true;
        }
        // footpedals config
        configFiles = jsonValue["footpedals"];
        if (!configFiles.empty()) {
            mHasIO = true;
        }
        // see if user wants to force no foot pedals
        Json::Value footpedalsRequired = jsonValue["physical-footpedals-required"];
        if (!footpedalsRequired.empty()) {
            physicalFootpedalsRequired = footpedalsRequired.asBool();
        }
    }
    // just check for IO and make sure we don't have io and hid, will be configured later
    jsonValue = jsonConfig["operator-present"];
    if (!jsonValue.empty()) {
        // check if operator present uses IO
        Json::Value jsonConfigFile = jsonValue["io"];
        if (!jsonConfigFile.empty()) {
            mHasIO = true;
            jsonConfigFile = jsonValue["hid"];
            if (!jsonConfigFile.empty()) {
                CMN_LOG_CLASS_INIT_ERROR << "Configure: operator-present can't have both io and hid" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    // create IO if needed and configure IO
    if (mHasIO) {
        mtsRobotIO1394 * io = new mtsRobotIO1394(m_IO_component_name, periodIO, port);
        io->SetProtocol(protocol);
        io->SetWatchdogPeriod(watchdogTimeout);
        // configure for each arm
        for (auto iter = mArms.begin(); iter != end; ++iter) {
            std::string ioConfig = iter->second->m_IO_configuration_file;
            if (ioConfig != "") {
                CMN_LOG_CLASS_INIT_VERBOSE << "Configure: configuring IO using \"" << ioConfig << "\"" << std::endl;
                io->Configure(ioConfig);
            }
            std::string ioGripperConfig = iter->second->m_IO_gripper_configuration_file;
            if (ioGripperConfig != "") {
                CMN_LOG_CLASS_INIT_VERBOSE << "Configure: configuring IO gripper using \"" << ioGripperConfig << "\"" << std::endl;
                io->Configure(ioGripperConfig);
            }
        }
        // configure using extra configuration files
        jsonValue = jsonConfig["io"];
        if (!jsonValue.empty()) {
            // generic files
            Json::Value configFiles = jsonValue["configuration-files"];
            if (!configFiles.empty()) {
                for (unsigned int index = 0; index < configFiles.size(); ++index) {
                    const std::string configFile = m_config_path.Find(configFiles[index].asString());
                    if (configFile == "") {
                        CMN_LOG_CLASS_INIT_ERROR << "Configure: can't find configuration file "
                                                 << configFiles[index].asString() << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    CMN_LOG_CLASS_INIT_VERBOSE << "Configure: configuring IO using \"" << configFile << "\"" << std::endl;
                    io->Configure(configFile);
                }
            }
            // footpedals, we assume these are the default one provided along the dVRK
            configFiles = jsonValue["footpedals"];
            if (!configFiles.empty()) {
                const std::string configFile = m_config_path.Find(configFiles.asString());
                if (configFile == "") {
                    CMN_LOG_CLASS_INIT_ERROR << "Configure: can't find configuration file "
                                             << configFiles.asString() << std::endl;
                    exit(EXIT_FAILURE);
                }
                CMN_LOG_CLASS_INIT_VERBOSE << "Configure: configuring IO foot pedals using \"" << configFile << "\"" << std::endl;
                // these can be overwritten using console-inputs
                mDInputSources["Clutch"] = InterfaceComponentType(m_IO_component_name, "Clutch");
                mDInputSources["OperatorPresent"] = InterfaceComponentType(m_IO_component_name, "Coag");
                mDInputSources["Coag"] = InterfaceComponentType(m_IO_component_name, "Coag");
                mDInputSources["BiCoag"] = InterfaceComponentType(m_IO_component_name, "BiCoag");
                mDInputSources["Camera"] = InterfaceComponentType(m_IO_component_name, "Camera");
                mDInputSources["Cam-"] = InterfaceComponentType(m_IO_component_name, "Cam-");
                mDInputSources["Cam+"] = InterfaceComponentType(m_IO_component_name, "Cam+");
                mDInputSources["Head"] = InterfaceComponentType(m_IO_component_name, "Head");
                io->Configure(configFile);
            }
            // check if user wants to close all relays
            Json::Value close_all_relays = jsonValue["close-all-relays"];
            if (!close_all_relays.empty()) {
                m_close_all_relays_from_config = close_all_relays.asBool();
            }
        }
        // configure IO for operator present
        jsonValue = jsonConfig["operator-present"];
        if (!jsonValue.empty()) {
            // check if operator present uses IO
            Json::Value jsonConfigFile = jsonValue["io"];
            if (!jsonConfigFile.empty()) {
                const std::string configFile = m_config_path.Find(jsonConfigFile.asString());
                if (configFile == "") {
                    CMN_LOG_CLASS_INIT_ERROR << "Configure: can't find configuration file "
                                             << jsonConfigFile.asString() << std::endl;
                    exit(EXIT_FAILURE);
                }
                CMN_LOG_CLASS_INIT_VERBOSE << "Configure: configuring operator present using \""
                                           << configFile << "\"" << std::endl;
                io->Configure(configFile);
            } else {
                jsonConfigFile = jsonValue["hid"];
            }
        }
        // configure for endoscope focus
        jsonValue = jsonConfig["endoscope-focus"];
        if (!jsonValue.empty()) {
            // check if operator present uses IO
            Json::Value jsonConfigFile = jsonValue["io"];
            if (!jsonConfigFile.empty()) {
                const std::string configFile = m_config_path.Find(jsonConfigFile.asString());
                if (configFile == "") {
                    CMN_LOG_CLASS_INIT_ERROR << "Configure: can't find configuration file "
                                             << jsonConfigFile.asString() << std::endl;
                    exit(EXIT_FAILURE);
                }
                CMN_LOG_CLASS_INIT_VERBOSE << "Configure: configuring endoscope focus using \""
                                           << configFile << "\"" << std::endl;
                io->Configure(configFile);
            }
        }
        // and add the io component!
        m_IO_interface = AddInterfaceRequired("IO");
        if (m_IO_interface) {
            m_IO_interface->AddFunction("close_all_relays", IO.close_all_relays);
            m_IO_interface->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::ErrorEventHandler,
                                                 this, "error");
            m_IO_interface->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::WarningEventHandler,
                                                 this, "warning");
            m_IO_interface->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::StatusEventHandler,
                                                 this, "status");
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to create IO required interface" << std::endl;
            exit(EXIT_FAILURE);
        }
        mtsComponentManager::GetInstance()->AddComponent(io);
        if (m_IO_interface) {
            mConnections.Add(this->GetName(), "IO",
                             io->GetName(), "Configuration");
        }
    }

    // now can configure PID and Arms
    for (auto iter = mArms.begin(); iter != end; ++iter) {
        auto arm_name = iter->first;
        auto arm_pointer = iter->second;
        // for generic arms, nothing to do
        if (arm_pointer->native_or_derived()
            || (arm_pointer->m_type == Arm::ARM_PSM_SOCKET)) {
            const std::string armConfig = arm_pointer->m_arm_configuration_file;
            arm_pointer->ConfigureArm(arm_pointer->m_type, armConfig,
                                      arm_pointer->m_arm_period);
        }
        // configure PID afterwards since we need the arm generation
        if (arm_pointer->expects_PID()) {
            // finally call configuration
            iter->second->ConfigurePID(arm_pointer->m_PID_configuration_file);
        }
    }

    // look for ECM teleop
    const Json::Value ecmTeleop = jsonConfig["ecm-teleop"];
    if (!ecmTeleop.isNull()) {
        if (!ConfigureECMTeleopJSON(ecmTeleop)) {
            CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to configure ecm-teleop" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // now load all PSM teleops
    const Json::Value psmTeleops = jsonConfig["psm-teleops"];
    for (unsigned int index = 0; index < psmTeleops.size(); ++index) {
        if (!ConfigurePSMTeleopJSON(psmTeleops[index])) {
            CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to configure psm-teleops[" << index << "]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // see which event is used for operator present
    // find name of button event used to detect if operator is present

    // load from console inputs
    const Json::Value consoleInputs = jsonConfig["console-inputs"];
    if (!consoleInputs.empty()) {
        std::string component, interface;
        component = consoleInputs["operator-present"]["component"].asString();
        interface = consoleInputs["operator-present"]["interface"].asString();
        if ((component != "") && (interface != "")) {
            mDInputSources["OperatorPresent"] = InterfaceComponentType(component, interface);
        }
        component = consoleInputs["clutch"]["component"].asString();
        interface = consoleInputs["clutch"]["interface"].asString();
        if ((component != "") && (interface != "")) {
            mDInputSources["Clutch"] = InterfaceComponentType(component, interface);
        }
        component = consoleInputs["camera"]["component"].asString();
        interface = consoleInputs["camera"]["interface"].asString();
        if ((component != "") && (interface != "")) {
            mDInputSources["Camera"] = InterfaceComponentType(component, interface);
        }
    }

    // load operator-present settings, this will over write older settings
    const Json::Value operatorPresent = jsonConfig["operator-present"];
    if (!operatorPresent.empty()) {
        // first case, using io to communicate with daVinci original head sensore
        Json::Value operatorPresentConfiguration = operatorPresent["io"];
        if (!operatorPresentConfiguration.empty()) {
            const std::string headSensorName = "daVinciHeadSensor";
            mHeadSensor = new mtsDaVinciHeadSensor(headSensorName);
            mtsComponentManager::GetInstance()->AddComponent(mHeadSensor);
            // main DInput is OperatorPresent comming from the newly added component
            mDInputSources["OperatorPresent"] = InterfaceComponentType(headSensorName, "OperatorPresent");
            // also expose the digital inputs from RobotIO (e.g. ROS topics)
            mDInputSources["HeadSensor1"] = InterfaceComponentType(m_IO_component_name, "HeadSensor1");
            mDInputSources["HeadSensor2"] = InterfaceComponentType(m_IO_component_name, "HeadSensor2");
            mDInputSources["HeadSensor3"] = InterfaceComponentType(m_IO_component_name, "HeadSensor3");
            mDInputSources["HeadSensor4"] = InterfaceComponentType(m_IO_component_name, "HeadSensor4");
            // schedule connections
            mConnections.Add(headSensorName, "HeadSensorTurnOff",
                             m_IO_component_name, "HeadSensorTurnOff");
            mConnections.Add(headSensorName, "HeadSensor1",
                             m_IO_component_name, "HeadSensor1");
            mConnections.Add(headSensorName, "HeadSensor2",
                             m_IO_component_name, "HeadSensor2");
            mConnections.Add(headSensorName, "HeadSensor3",
                             m_IO_component_name, "HeadSensor3");
            mConnections.Add(headSensorName, "HeadSensor4",
                             m_IO_component_name, "HeadSensor4");
        } else {
            // second case, using hid config for goovis head sensor
            operatorPresentConfiguration = operatorPresent["hid"];
            if (!operatorPresentConfiguration.empty()) {
#if sawIntuitiveResearchKit_HAS_HID_HEAD_SENSOR
                std::string relativeConfigFile = operatorPresentConfiguration.asString();
                CMN_LOG_CLASS_INIT_VERBOSE << "Configure: configuring hid head sensor with \""
                                           << relativeConfigFile << "\"" << std::endl;
                const std::string configFile = m_config_path.Find(relativeConfigFile);
                if (configFile == "") {
                    CMN_LOG_CLASS_INIT_ERROR << "Configure: can't find configuration file "
                                             << relativeConfigFile << std::endl;
                    exit(EXIT_FAILURE);
                }
                const std::string headSensorName = "HIDHeadSensor";
                mHeadSensor = new mtsHIDHeadSensor(headSensorName);
                mHeadSensor->Configure(configFile);
                mtsComponentManager::GetInstance()->AddComponent(mHeadSensor);
                // main DInput is OperatorPresent comming from the newly added component
                mDInputSources["OperatorPresent"] = InterfaceComponentType(headSensorName, "OperatorPresent");
#else
                CMN_LOG_CLASS_INIT_ERROR << "Configure: can't use HID head sensor." << std::endl
                                         << "The code has been compiled with sawIntuitiveResearchKit_HAS_HID_HEAD_SENSOR OFF." << std::endl
                                         << "Re-run CMake, re-compile and try again." << std::endl;
                exit(EXIT_FAILURE);
#endif
            }
        }
    }

    // message re. footpedals are likely missing but user can override this requirement
    const std::string footpedalMessage = "Maybe you're missing \"io\":\"footpedals\" in your configuration file.  If you don't need physical footpedals, set \"physical-footpedals-required\" to false.";

    // load endoscope-focus settings
    const Json::Value endoscopeFocus = jsonConfig["endoscope-focus"];
    if (!endoscopeFocus.empty()) {
        const std::string endoscopeFocusName = "daVinciEndoscopeFocus";
        mDaVinciEndoscopeFocus = new mtsDaVinciEndoscopeFocus(endoscopeFocusName);
        mtsComponentManager::GetInstance()->AddComponent(mDaVinciEndoscopeFocus);
        // make sure we have cam+ and cam- in digital inputs
        const DInputSourceType::const_iterator endDInputs = mDInputSources.end();
        const bool foundCamMinus = (mDInputSources.find("Cam-") != endDInputs);
        if (!foundCamMinus) {
            CMN_LOG_CLASS_INIT_ERROR << "Configure: input for footpedal \"Cam-\" is required for \"endoscope-focus\".  "
                                     << footpedalMessage << std::endl;
            exit(EXIT_FAILURE);
        }
        const bool foundCamPlus = (mDInputSources.find("Cam+") != endDInputs);
        if (!foundCamPlus) {
            CMN_LOG_CLASS_INIT_ERROR << "Configure: input for footpedal \"Cam+\" is required for \"endoscope-focus\".  "
                                     << footpedalMessage << std::endl;
            exit(EXIT_FAILURE);
        }
        // schedule connections
        mConnections.Add(endoscopeFocusName, "EndoscopeFocusIn",
                         m_IO_component_name, "EndoscopeFocusIn");
        mConnections.Add(endoscopeFocusName, "EndoscopeFocusOut",
                         m_IO_component_name, "EndoscopeFocusOut");
        mConnections.Add(endoscopeFocusName, "focus_in",
                         m_IO_component_name, "Cam+");
        mConnections.Add(endoscopeFocusName, "focus_out",
                         m_IO_component_name, "Cam-");
    }

    // if we have any teleoperation component, we need to have the interfaces for the foot pedals
    // unless user explicitly says we can skip
    if (physicalFootpedalsRequired) {
        const DInputSourceType::const_iterator endDInputs = mDInputSources.end();
        const bool foundClutch = (mDInputSources.find("Clutch") != endDInputs);
        const bool foundOperatorPresent = (mDInputSources.find("OperatorPresent") != endDInputs);
        const bool foundCamera = (mDInputSources.find("Camera") != endDInputs);

        if (mTeleopsPSM.size() > 0) {
            if (!foundClutch || !foundOperatorPresent) {
                CMN_LOG_CLASS_INIT_ERROR << "Configure: inputs for footpedals \"Clutch\" and \"OperatorPresent\" need to be defined since there's at least one PSM tele-operation component.  "
                                         << footpedalMessage << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (mTeleopECM) {
            if (!foundCamera || !foundOperatorPresent) {
                CMN_LOG_CLASS_INIT_ERROR << "Configure: inputs for footpedals \"Camera\" and \"OperatorPresent\" need to be defined since there's an ECM tele-operation component.  "
                                         << footpedalMessage << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    this->AddFootpedalInterfaces();

    // search for SUJs, real, not Fixed
    bool hasSUJ = false;
    for (auto iter = mArms.begin(); iter != end; ++iter) {
        if ((iter->second->m_type == Arm::ARM_SUJ_Classic)
            || (iter->second->m_type == Arm::ARM_SUJ_Si)) {
            hasSUJ = true;
        }
    }

    if (hasSUJ) {
        for (auto iter = mArms.begin(); iter != end; ++iter) {
            Arm * arm = iter->second;
            // only for PSM and ECM when not simulated
            if (((arm->m_type == Arm::ARM_ECM)
                 || (arm->m_type == Arm::ARM_ECM_DERIVED)
                 || (arm->m_type == Arm::ARM_PSM)
                 || (arm->m_type == Arm::ARM_PSM_DERIVED)
                 )
                && (arm->m_simulation == Arm::SIMULATION_NONE)) {
                arm->SUJInterfaceRequiredFromIO = this->AddInterfaceRequired("SUJClutch-" + arm->Name() + "-IO");
                arm->SUJInterfaceRequiredFromIO->AddEventHandlerWrite(&Arm::SUJClutchEventHandlerFromIO, arm, "Button");
                if (arm->m_generation == mtsIntuitiveResearchKitArm::GENERATION_Si) {
                    arm->SUJInterfaceRequiredFromIO2 = this->AddInterfaceRequired("SUJClutchBack-" + arm->Name() + "-IO");
                    arm->SUJInterfaceRequiredFromIO2->AddEventHandlerWrite(&Arm::SUJClutchEventHandlerFromIO, arm, "Button");
                }
                arm->SUJInterfaceRequiredToSUJ = this->AddInterfaceRequired("SUJClutch-" + arm->Name());
                arm->SUJInterfaceRequiredToSUJ->AddFunction("Clutch", arm->SUJClutch);
            }
        }
    }

    m_configured = true;
}

const bool & mtsIntuitiveResearchKitConsole::Configured(void) const
{
    return m_configured;
}

void mtsIntuitiveResearchKitConsole::Startup(void)
{
    std::string message = this->GetName();
    message.append(" started, dVRK ");
    message.append(sawIntuitiveResearchKit_VERSION);
    message.append(" / cisst ");
    message.append(cisst_VERSION);
    mInterface->SendStatus(message);

    // close all relays if needed
    if (m_close_all_relays_from_config) {
        IO.close_all_relays();
    }

    // emit events for active PSM teleop pairs
    EventSelectedTeleopPSMs();
    // emit scale event
    ConfigurationEvents.scale(mtsIntuitiveResearchKit::TeleOperationPSM::Scale);
    // emit volume event
    audio.volume(m_audio_volume);

    if (mChatty) {
        // someone is going to hate me for this :-)
        std::vector<std::string> prompts;
        prompts.push_back("Hello");
        prompts.push_back("It will not work!");
        prompts.push_back("It might work");
        prompts.push_back("It will work!");
        prompts.push_back("Are we there yet?");
        prompts.push_back("When is that paper deadline?");
        prompts.push_back("Don't you have something better to do?");
        prompts.push_back("Today is the day!");
        prompts.push_back("It's free software, what did you expect?");
        prompts.push_back("I didn't do it!");
        prompts.push_back("Be careful!");
        prompts.push_back("Peter will fix it");
        prompts.push_back("Ask Google");
        prompts.push_back("Did you forget to re-compile?");
        prompts.push_back("Reboot me");
        prompts.push_back("Coffee break?");
        prompts.push_back("If you can hear this, the code compiles!");
        prompts.push_back("I'm a bit tired");
        prompts.push_back("Commit often, always pull!");
        prompts.push_back("Feel free to fix it");
        prompts.push_back("What's the weather like outside?");
        prompts.push_back("Call your parents");
        prompts.push_back("Maybe we should use ROS control");
        prompts.push_back("Use with caution");
        prompts.push_back("It's about time");
        prompts.push_back("When did you last commit your changes?");
        prompts.push_back("Some documentation would be nice");
        int index;
        cmnRandomSequence & randomSequence = cmnRandomSequence::GetInstance();
        cmnRandomSequence::SeedType seed
            = static_cast<cmnRandomSequence::SeedType>(mtsManagerLocal::GetInstance()->GetTimeServer().GetRelativeTime() * 100000.0);
        randomSequence.SetSeed(seed % 1000);
        randomSequence.ExtractRandomValue<int>(0, prompts.size() - 1, index);
        audio.string_to_speech(prompts.at(index));
    }
}

void mtsIntuitiveResearchKitConsole::Run(void)
{
    ProcessQueuedCommands();
    ProcessQueuedEvents();
}

void mtsIntuitiveResearchKitConsole::Cleanup(void)
{
    CMN_LOG_CLASS_INIT_VERBOSE << "Cleanup" << std::endl;
}

bool mtsIntuitiveResearchKitConsole::AddArm(Arm * newArm)
{
    if (AddArmInterfaces(newArm)) {
        auto armIterator = mArms.find(newArm->m_name);
        if (armIterator == mArms.end()) {
            mArms[newArm->m_name] = newArm;
            return true;
        } else {
            CMN_LOG_CLASS_INIT_ERROR << GetName() << ": AddArm, "
                                     << newArm->Name() << " seems to already exist (Arm config)." << std::endl;
        }
    }
    CMN_LOG_CLASS_INIT_ERROR << GetName() << ": AddArm, unable to add new arm.  Are you adding two arms with the same name? "
                             << newArm->Name() << std::endl;
    return false;
}

bool mtsIntuitiveResearchKitConsole::AddArm(mtsComponent * genericArm, const mtsIntuitiveResearchKitConsole::Arm::ArmType CMN_UNUSED(arm_type))
{
    // create new required interfaces to communicate with the components we created
    Arm * newArm = new Arm(this, genericArm->GetName(), "");
    if (AddArmInterfaces(newArm)) {
        auto armIterator = mArms.find(newArm->m_name);
        if (armIterator != mArms.end()) {
            mArms[newArm->m_name] = newArm;
            return true;
        }
    }
    CMN_LOG_CLASS_INIT_ERROR << GetName() << ": AddArm, unable to add new arm.  Are you adding two arms with the same name? "
                             << newArm->Name() << std::endl;
    delete newArm;
    return false;
}

std::string mtsIntuitiveResearchKitConsole::GetArmIOComponentName(const std::string & arm_name)
{
    auto armIterator = mArms.find(arm_name);
    if (armIterator != mArms.end()) {
        return armIterator->second->m_IO_component_name;
    }
    return "";
}

bool mtsIntuitiveResearchKitConsole::AddTeleopECMInterfaces(TeleopECM * teleop)
{
    teleop->InterfaceRequired = this->AddInterfaceRequired(teleop->Name());
    if (teleop->InterfaceRequired) {
        teleop->InterfaceRequired->AddFunction("state_command", teleop->state_command);
        teleop->InterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::ErrorEventHandler, this, "error");
        teleop->InterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::WarningEventHandler, this, "warning");
        teleop->InterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::StatusEventHandler, this, "status");
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddTeleopECMInterfaces: failed to add Main interface for teleop \""
                                 << teleop->Name() << "\"" << std::endl;
        return false;
    }
    return true;
}

bool mtsIntuitiveResearchKitConsole::AddTeleopPSMInterfaces(TeleopPSM * teleop)
{
    teleop->InterfaceRequired = this->AddInterfaceRequired(teleop->Name());
    if (teleop->InterfaceRequired) {
        teleop->InterfaceRequired->AddFunction("state_command", teleop->state_command);
        teleop->InterfaceRequired->AddFunction("set_scale", teleop->set_scale);
        teleop->InterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::ErrorEventHandler, this, "error");
        teleop->InterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::WarningEventHandler, this, "warning");
        teleop->InterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::StatusEventHandler, this, "status");
        teleop->InterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::TeleopScaleChangedEventHandler, this, "scale");
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddTeleopPSMInterfaces: failed to add Main interface for teleop \""
                                 << teleop->Name() << "\"" << std::endl;
        return false;
    }
    return true;
}

void mtsIntuitiveResearchKitConsole::AddFootpedalInterfaces(void)
{
    const auto endDInputs = mDInputSources.end();

    auto iter = mDInputSources.find("Clutch");
    if (iter != endDInputs) {
        mtsInterfaceRequired * clutchRequired = AddInterfaceRequired("Clutch");
        if (clutchRequired) {
            clutchRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::ClutchEventHandler, this, "Button");
        }
        mConnections.Add(this->GetName(), "Clutch",
                         iter->second.first, iter->second.second);
    }
    mtsInterfaceProvided * clutchProvided = AddInterfaceProvided("Clutch");
    if (clutchProvided) {
        clutchProvided->AddEventWrite(console_events.clutch, "Button", prmEventButton());
    }

    iter = mDInputSources.find("Camera");
    if (iter != endDInputs) {
        mtsInterfaceRequired * cameraRequired = AddInterfaceRequired("Camera");
        if (cameraRequired) {
            cameraRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::CameraEventHandler, this, "Button");
        }
        mConnections.Add(this->GetName(), "Camera",
                         iter->second.first, iter->second.second);
    }
    mtsInterfaceProvided * cameraProvided = AddInterfaceProvided("Camera");
    if (cameraProvided) {
        cameraProvided->AddEventWrite(console_events.camera, "Button", prmEventButton());
    }

    iter = mDInputSources.find("OperatorPresent");
    if (iter != endDInputs) {
        mtsInterfaceRequired * operatorRequired = AddInterfaceRequired("OperatorPresent");
        if (operatorRequired) {
            operatorRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::OperatorPresentEventHandler, this, "Button");
        }
        mConnections.Add(this->GetName(), "OperatorPresent",
                         iter->second.first, iter->second.second);
    }
    mtsInterfaceProvided * operatorProvided = AddInterfaceProvided("OperatorPresent");
    if (operatorProvided) {
        operatorProvided->AddEventWrite(console_events.operator_present, "Button", prmEventButton());
    }
}

bool mtsIntuitiveResearchKitConsole::ConfigureArmJSON(const Json::Value & jsonArm,
                                                      const std::string & ioComponentName)                                          {
    const std::string arm_name = jsonArm["name"].asString();
    const auto armIterator = mArms.find(arm_name);
    Arm * arm_pointer = 0;
    if (armIterator == mArms.end()) {
        // create a new arm if needed
        arm_pointer = new Arm(this, arm_name, ioComponentName);
    } else {
        arm_pointer = armIterator->second;
    }

    Json::Value jsonValue;

    // create search path based on optional system
    arm_pointer->m_config_path = m_config_path;
    jsonValue = jsonArm["system"];
    if (!jsonValue.empty()) {
        arm_pointer->m_config_path.Add(std::string(sawIntuitiveResearchKit_SOURCE_DIR)
                                       + "/../share/"
                                       + jsonValue.asString() + "/",
                                       cmnPath::TAIL);
    }

    // read from JSON and check if configuration files exist
    jsonValue = jsonArm["type"];
    if (!jsonValue.empty()) {
        std::string typeString = jsonValue.asString();
        if (typeString == "MTM") {
            arm_pointer->m_type = Arm::ARM_MTM;
        } else if (typeString == "PSM") {
            arm_pointer->m_type = Arm::ARM_PSM;
        } else if (typeString == "ECM") {
            arm_pointer->m_type = Arm::ARM_ECM;
        } else if (typeString == "MTM_DERIVED") {
            arm_pointer->m_type = Arm::ARM_MTM_DERIVED;
        } else if (typeString == "PSM_DERIVED") {
            arm_pointer->m_type = Arm::ARM_PSM_DERIVED;
        } else if (typeString == "ECM_DERIVED") {
            arm_pointer->m_type = Arm::ARM_ECM_DERIVED;
        } else if (typeString == "MTM_GENERIC") {
            arm_pointer->m_type = Arm::ARM_MTM_GENERIC;
        } else if (typeString == "PSM_GENERIC") {
            arm_pointer->m_type = Arm::ARM_PSM_GENERIC;
        } else if (typeString == "ECM_GENERIC") {
            arm_pointer->m_type = Arm::ARM_ECM_GENERIC;
        } else if (typeString == "PSM_SOCKET") {
            arm_pointer->m_type = Arm::ARM_PSM_SOCKET;
        } else if (typeString == "FOCUS_CONTROLLER") {
            arm_pointer->m_type = Arm::FOCUS_CONTROLLER;
        } else if (typeString == "SUJ_Classic") {
            arm_pointer->m_type = Arm::ARM_SUJ_Classic;
        } else if (typeString == "SUJ_Si") {
            arm_pointer->m_type = Arm::ARM_SUJ_Si;
        } else if (typeString == "SUJ_Fixed") {
            arm_pointer->m_type = Arm::ARM_SUJ_Fixed;
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: arm " << arm_name << ": invalid type \""
                                     << typeString << "\", needs to be one of {MTM,PSM,ECM}{,_DERIVED,_GENERIC} or SUJ_{Classic,Si,Fixed}" << std::endl;
            return false;
        }
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: arm " << arm_name
                                 << ": doesn't have a \"type\" specified, needs to be one of {MTM,PSM,ECM,SUJ}{,_DERIVED,_GENERIC} or SUJ_{Classic,Si}" << std::endl;
        return false;
    }

    jsonValue = jsonArm["serial"];
    if (!jsonValue.empty()) {
        arm_pointer->m_serial = jsonValue.asString();
    }

    // type of simulation, if any
    jsonValue = jsonArm["simulation"];
    if (!jsonValue.empty()) {
        std::string typeString = jsonValue.asString();
        if (typeString == "KINEMATIC") {
            arm_pointer->m_simulation = Arm::SIMULATION_KINEMATIC;
        } else if (typeString == "DYNAMIC") {
            arm_pointer->m_simulation = Arm::SIMULATION_DYNAMIC;
        } else if (typeString == "NONE") {
            arm_pointer->m_simulation = Arm::SIMULATION_NONE;
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: arm " << arm_name << ": invalid simulation \""
                                     << typeString << "\", needs to be NONE, KINEMATIC or DYNAMIC" << std::endl;
            return false;
        }
    } else {
        arm_pointer->m_simulation = Arm::SIMULATION_NONE;
    }

    // set arm calibration mode based on console calibration mode
    arm_pointer->m_calibration_mode = m_calibration_mode;

    // should we automatically create ROS bridge for this arm
    arm_pointer->m_skip_ROS_bridge = false;
    jsonValue = jsonArm["skip-ros-bridge"];
    if (!jsonValue.empty()) {
        arm_pointer->m_skip_ROS_bridge = jsonValue.asBool();
    }

    // component and interface, defaults
    arm_pointer->m_arm_component_name = arm_name;
    arm_pointer->m_arm_interface_name = "Arm";
    jsonValue = jsonArm["component"];
    if (!jsonValue.empty()) {
        arm_pointer->m_arm_component_name = jsonValue.asString();
    }
    jsonValue = jsonArm["interface"];
    if (!jsonValue.empty()) {
        arm_pointer->m_arm_interface_name = jsonValue.asString();
    }

    // check if we need to create a socket server attached to this arm
    arm_pointer->m_socket_server = false;
    jsonValue = jsonArm["socket-server"];
    if (!jsonValue.empty()) {
        arm_pointer->m_socket_server = jsonValue.asBool();
    }

    // for socket client or server, look for remote IP / port
    if (arm_pointer->m_type == Arm::ARM_PSM_SOCKET || arm_pointer->m_socket_server) {
        arm_pointer->m_socket_component_name = arm_pointer->m_name + "-SocketServer";
        jsonValue = jsonArm["remote-ip"];
        if(!jsonValue.empty()){
            arm_pointer->m_IP = jsonValue.asString();
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find \"server-ip\" for arm \""
                                     << arm_name << "\"" << std::endl;
            return false;
        }
        jsonValue = jsonArm["port"];
        if (!jsonValue.empty()) {
            arm_pointer->m_port = jsonValue.asInt();
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find \"port\" for arm \""
                                     << arm_name << "\"" << std::endl;
            return false;
        }
    }

    // IO for anything not simulated or socket client or Si SUJ
    if (arm_pointer->expects_IO()) {
        jsonValue = jsonArm["io"];
        if (!jsonValue.empty()) {
            arm_pointer->m_IO_configuration_file = arm_pointer->m_config_path.Find(jsonValue.asString());
            if (arm_pointer->m_IO_configuration_file == "") {
                CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find IO file " << jsonValue.asString() << std::endl;
                return false;
            }
        } else {
            // try to find default if serial number has been provided
            if (arm_pointer->m_serial != "") {
                std::string defaultFile = "sawRobotIO1394-" + arm_name + "-" + arm_pointer->m_serial + ".xml";
                arm_pointer->m_IO_configuration_file = arm_pointer->m_config_path.Find(defaultFile);
                if (arm_pointer->m_IO_configuration_file == "") {
                    CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find IO file " << defaultFile << std::endl;
                    return false;
                }
            } else {
                // no io nor serial
                CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find \"io\" setting for arm \""
                                         << arm_name << "\" and \"serial\" is not provided so we can't search for it" << std::endl;
                return false;
            }
        }
        // IO for MTM gripper
        if ((arm_pointer->m_type == Arm::ARM_MTM)
            || (arm_pointer->m_type == Arm::ARM_MTM_DERIVED)) {
            jsonValue = jsonArm["io-gripper"];
            if (!jsonValue.empty()) {
                arm_pointer->m_IO_gripper_configuration_file = arm_pointer->m_config_path.Find(jsonValue.asString());
                if (arm_pointer->m_IO_gripper_configuration_file == "") {
                    CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find IO gripper file "
                                             << jsonValue.asString() << std::endl;
                    return false;
                }
            } else {
                // try to find default if serial number has been provided
                if (arm_pointer->m_serial != "") {
                    std::string defaultFile = "sawRobotIO1394-" + arm_name + "-gripper-" + arm_pointer->m_serial + ".xml";
                    arm_pointer->m_IO_gripper_configuration_file = arm_pointer->m_config_path.Find(defaultFile);
                    if (arm_pointer->m_IO_gripper_configuration_file == "") {
                        CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find IO gripper file " << defaultFile << std::endl;
                        return false;
                    }
                } else {
                    // no io nor serial
                    CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find \"io-gripper\" setting for arm \""
                                             << arm_name << "\" and \"serial\" is not provided so we can't search for it" << std::endl;
                    return false;
                }
            }
        }
    }

    // PID only required for MTM, PSM and ECM (and derived)
    if (arm_pointer->expects_PID()) {
        jsonValue = jsonArm["pid"];
        if (!jsonValue.empty()) {
            arm_pointer->m_PID_configuration_file = arm_pointer->m_config_path.Find(jsonValue.asString());
            if (arm_pointer->m_PID_configuration_file == "") {
                CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find PID file " << jsonValue.asString() << std::endl;
                return false;
            }
        }
    }

    // only configure kinematics if not arm socket client
    if ((arm_pointer->m_type != Arm::ARM_PSM_SOCKET)
        && (arm_pointer->native_or_derived())) {
        // renamed "kinematic" to "arm" so we can have a more complex configuration file for the arm class
        jsonValue = jsonArm["arm"];
        if (!jsonValue.empty()) {
            arm_pointer->m_arm_configuration_file = arm_pointer->m_config_path.Find(jsonValue.asString());
            if (arm_pointer->m_arm_configuration_file == "") {
                CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find configuration file " << jsonValue.asString() << std::endl;
                return false;
            }
        }
        jsonValue = jsonArm["kinematic"];
        if (!jsonValue.empty()) {
            if (arm_pointer->m_arm_configuration_file != "") {
                CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: arm configuration file is already set using \"arm\", you should remove the deprecated \"kinematic\" field:"
                                         << jsonValue.asString() << std::endl;
                return false;
            } else {
                arm_pointer->m_arm_configuration_file = arm_pointer->m_config_path.Find(jsonValue.asString());
                if (arm_pointer->m_arm_configuration_file == "") {
                    CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find arm configuration file " << jsonValue.asString() << std::endl;
                    return false;
                }
            }
        }

        // make sure we have an arm configuration file for all arms except FOCUS_CONTROLLER
        if ((arm_pointer->m_arm_configuration_file == "")
            && (arm_pointer->m_type != Arm::FOCUS_CONTROLLER)) {
            if (arm_pointer->native_or_derived()) {
                // try to find the arm file using default
                std::string defaultFile = arm_name + "-" + arm_pointer->m_serial + ".json";
                arm_pointer->m_arm_configuration_file = arm_pointer->m_config_path.Find(defaultFile);
                if (arm_pointer->m_arm_configuration_file == "") {
                    CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find \"arm\" setting for arm \""
                                             << arm_name << "\".  \"arm\" is not set and the default file \""
                                             << defaultFile << "\" doesn't seem to exist either." << std::endl;
                    return false;
                }
            } else {
                CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: can't find \"kinematic\" setting for arm \""
                                         << arm_name << "\"" << std::endl;
                return false;
            }
        }

        jsonValue = jsonArm["base-frame"];
        if (!jsonValue.empty()) {
            Json::Value fixedJson = jsonValue["transform"];
            if (!fixedJson.empty()) {
                std::string reference = jsonValue["reference-frame"].asString();
                if (reference.empty()) {
                    CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: both \"transform\" (4x4) and \"reference-frame\" (name) must be provided with \"base-frame\" for arm \""
                                             << arm_name << "\"" << std::endl;
                    return false;
                }
                vctFrm4x4 frame;
                cmnDataJSON<vctFrm4x4>::DeSerializeText(frame, fixedJson);
                arm_pointer->m_base_frame.Goal().From(frame);
                arm_pointer->m_base_frame.ReferenceFrame() = reference;
                arm_pointer->m_base_frame.Valid() = true;
            } else {
                arm_pointer->m_base_frame_component_name = jsonValue.get("component", "").asString();
                arm_pointer->m_base_frame_interface_name = jsonValue.get("interface", "").asString();
                if ((arm_pointer->m_base_frame_component_name == "")
                    || (arm_pointer->m_base_frame_interface_name == "")) {
                    CMN_LOG_CLASS_INIT_ERROR << "ConfigureArmJSON: both \"component\" and \"interface\" OR \"transform\" (4x4) and \"reference-frame\" (name) must be provided with \"base-frame\" for arm \""
                                             << arm_name << "\"" << std::endl;
                    return false;
                }
            }
        }
    }

    // read period if present
    jsonValue = jsonArm["period"];
    if (!jsonValue.empty()) {
        arm_pointer->m_arm_period = jsonValue.asFloat();
    }

    // add the arm if it's a new one
    if (armIterator == mArms.end()) {
        AddArm(arm_pointer);
    }
    return true;
}

bool mtsIntuitiveResearchKitConsole::ConfigureECMTeleopJSON(const Json::Value & jsonTeleop)
{
    std::string mtmLeftName = jsonTeleop["mtm-left"].asString();
    // for backward compatibility
    if (mtmLeftName == "") {
        mtmLeftName = jsonTeleop["master-left"].asString();
        CMN_LOG_CLASS_INIT_WARNING << "ConfigureECMTeleopJSON: keyword \"master-left\" is deprecated, use \"mtm-left\" instead" << std::endl;
    }
    std::string mtmRightName = jsonTeleop["mtm-right"].asString();
    // for backward compatibility
    if (mtmRightName == "") {
        mtmRightName = jsonTeleop["master-right"].asString();
        CMN_LOG_CLASS_INIT_WARNING << "ConfigureECMTeleopJSON: keyword \"master-right\" is deprecated, use \"mtm-right\" instead" << std::endl;
    }
    std::string ecmName = jsonTeleop["ecm"].asString();
    // for backward compatibility
    if (ecmName == "") {
        ecmName = jsonTeleop["slave"].asString();
        CMN_LOG_CLASS_INIT_WARNING << "ConfigureECMTeleopJSON: keyword \"slave\" is deprecated, use \"ecm\" instead" << std::endl;
    }
    // all must be provided
    if ((mtmLeftName == "") || (mtmRightName == "") || (ecmName == "")) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: \"mtm-left\", \"mtm-right\" and \"ecm\" must be provided as strings" << std::endl;
        return false;
    }

    if (mtmLeftName == mtmRightName) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: \"mtm-left\" and \"mtm-right\" must be different" << std::endl;
        return false;
    }
    std::string
        mtmLeftComponent, mtmLeftInterface,
        mtmRightComponent, mtmRightInterface,
        ecmComponent, ecmInterface;
    // check that both arms have been defined and have correct type
    Arm * arm_pointer;
    auto armIterator = mArms.find(mtmLeftName);
    if (armIterator == mArms.end()) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: mtm left\""
                                 << mtmLeftName << "\" is not defined in \"arms\"" << std::endl;
        return false;
    } else {
        arm_pointer = armIterator->second;
        if (!arm_pointer->mtm()) {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: mtm left\""
                                     << mtmLeftName << "\" type must be some kind of MTM" << std::endl;
            return false;
        }
        mtmLeftComponent = arm_pointer->ComponentName();
        mtmLeftInterface = arm_pointer->InterfaceName();
    }
    armIterator = mArms.find(mtmRightName);
    if (armIterator == mArms.end()) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: mtm right\""
                                 << mtmRightName << "\" is not defined in \"arms\"" << std::endl;
        return false;
    } else {
        arm_pointer = armIterator->second;
        if (!arm_pointer->mtm()) {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: mtm right\""
                                     << mtmRightName << "\" type must be some kind of MTM" << std::endl;
            return false;
        }
        mtmRightComponent = arm_pointer->ComponentName();
        mtmRightInterface = arm_pointer->InterfaceName();
    }
    armIterator = mArms.find(ecmName);
    if (armIterator == mArms.end()) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: ecm \""
                                 << ecmName << "\" is not defined in \"arms\"" << std::endl;
        return false;
    } else {
        arm_pointer = armIterator->second;
        if (!arm_pointer->ecm()) {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: ecm \""
                                     << ecmName << "\" type must be some kind of ECM" << std::endl;
            return false;
        }
        ecmComponent = arm_pointer->ComponentName();
        ecmInterface = arm_pointer->InterfaceName();
    }

    // check if pair already exist and then add
    const std::string name = mtmLeftName + "-" + mtmRightName + "-" + ecmName;
    if (mTeleopECM == 0) {
        // create a new teleop if needed
        mTeleopECM = new TeleopECM(name);
        // schedule connections
        mConnections.Add(name, "MTML", mtmLeftComponent, mtmLeftInterface);
        mConnections.Add(name, "MTMR", mtmRightComponent, mtmRightInterface);
        mConnections.Add(name, "ECM", ecmComponent, ecmInterface);
        mConnections.Add(name, "Clutch", this->GetName(), "Clutch"); // console clutch
        mConnections.Add(this->GetName(), name, name, "Setting");
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: there is already an ECM teleop" << std::endl;
        return false;
    }

    Json::Value jsonValue;
    jsonValue = jsonTeleop["type"];
    if (!jsonValue.empty()) {
        std::string typeString = jsonValue.asString();
        if (typeString == "TELEOP_ECM") {
            mTeleopECM->m_type = TeleopECM::TELEOP_ECM;
        } else if (typeString == "TELEOP_ECM_DERIVED") {
            mTeleopECM->m_type = TeleopECM::TELEOP_ECM_DERIVED;
        } else if (typeString == "TELEOP_ECM_GENERIC") {
            mTeleopECM->m_type = TeleopECM::TELEOP_ECM_GENERIC;
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: teleop " << name << ": invalid type \""
                                     << typeString << "\", needs to be TELEOP_ECM, TELEOP_ECM_DERIVED or TELEOP_ECM_GENERIC" << std::endl;
            return false;
        }
    } else {
        // default value
        mTeleopECM->m_type = TeleopECM::TELEOP_ECM;
    }

    // read period if present
    double period = mtsIntuitiveResearchKit::TeleopPeriod;
    jsonValue = jsonTeleop["period"];
    if (!jsonValue.empty()) {
        period = jsonValue.asFloat();
    }
    // for backward compatibility, send warning
    jsonValue = jsonTeleop["rotation"];
    if (!jsonValue.empty()) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigureECMTeleopJSON: teleop " << name << ": \"rotation\" must now be defined under \"configure-parameter\" or in a separate configuration file" << std::endl;
        return false;
    }
    const Json::Value jsonTeleopConfig = jsonTeleop["configure-parameter"];
    mTeleopECM->ConfigureTeleop(mTeleopECM->m_type, period, jsonTeleopConfig);
    AddTeleopECMInterfaces(mTeleopECM);
    return true;
}

bool mtsIntuitiveResearchKitConsole::ConfigurePSMTeleopJSON(const Json::Value & jsonTeleop)
{
    std::string mtmName = jsonTeleop["mtm"].asString();
    // for backward compatibility
    if (mtmName == "") {
        mtmName = jsonTeleop["master"].asString();
        CMN_LOG_CLASS_INIT_WARNING << "ConfigurePSMTeleopJSON: keyword \"master\" is deprecated, use \"mtm\" instead" << std::endl;
    }
    std::string psmName = jsonTeleop["psm"].asString();
    // for backward compatibility
    if (psmName == "") {
        psmName = jsonTeleop["slave"].asString();
        CMN_LOG_CLASS_INIT_WARNING << "ConfigurePSMTeleopJSON: keyword \"slave\" is deprecated, use \"psm\" instead" << std::endl;
    }
    // both are required
    if ((mtmName == "") || (psmName == "")) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigurePSMTeleopJSON: both \"mtm\" and \"psm\" must be provided as strings" << std::endl;
        return false;
    }

    std::string mtmComponent, mtmInterface, psmComponent, psmInterface;
    // check that both arms have been defined and have correct type
    Arm * arm_pointer;
    auto armIterator = mArms.find(mtmName);
    if (armIterator == mArms.end()) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigurePSMTeleopJSON: mtm \""
                                 << mtmName << "\" is not defined in \"arms\"" << std::endl;
        return false;
    } else {
        arm_pointer = armIterator->second;
        if (!arm_pointer->mtm()) {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigurePSMTeleopJSON: mtm \""
                                     << mtmName << "\" type must be some kind of MTM" << std::endl;
            return false;
        }
        mtmComponent = arm_pointer->ComponentName();
        mtmInterface = arm_pointer->InterfaceName();
    }
    armIterator = mArms.find(psmName);
    if (armIterator == mArms.end()) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigurePSMTeleopJSON: psm \""
                                 << psmName << "\" is not defined in \"arms\"" << std::endl;
        return false;
    } else {
        arm_pointer = armIterator->second;
        if (!arm_pointer->psm()) {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigurePSMTeleopJSON: psm \""
                                     << psmName << "\" type must be some kind of PSM" << std::endl;
            return false;
        }
        psmComponent = arm_pointer->ComponentName();
        psmInterface = arm_pointer->InterfaceName();
    }

    // see if there is a base frame defined for the PSM
    Json::Value jsonValue = jsonTeleop["psm-base-frame"];
    std::string baseFrameComponent, baseFrameInterface;
    if (!jsonValue.empty()) {
        baseFrameComponent = jsonValue.get("component", "").asString();
        baseFrameInterface = jsonValue.get("interface", "").asString();
        if ((baseFrameComponent == "") || (baseFrameInterface == "")) {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigurePSMTeleopJSON: both \"component\" and \"interface\" must be provided with \"psm-base-frame\" for teleop \""
                                     << mtmName << "-" << psmName << "\"" << std::endl;
            return false;
        }
    }

    // check if pair already exist and then add
    const std::string name = mtmName + "-" + psmName;
    const auto teleopIterator = mTeleopsPSM.find(name);
    TeleopPSM * teleopPointer = 0;
    if (teleopIterator == mTeleopsPSM.end()) {
        // create a new teleop if needed
        teleopPointer = new TeleopPSM(name, mtmName, psmName);
        // schedule connections
        mConnections.Add(name, "MTM", mtmComponent, mtmInterface);
        mConnections.Add(name, "PSM", psmComponent, psmInterface);
        mConnections.Add(name, "Clutch", this->GetName(), "Clutch"); // clutch from console
        mConnections.Add(this->GetName(), name, name, "Setting");
        if ((baseFrameComponent != "")
            && (baseFrameInterface != "")) {
            mConnections.Add(name, "PSM-base-frame",
                             baseFrameComponent, baseFrameInterface);
        }

        // insert
        mTeleopsPSMByMTM.insert(std::make_pair(mtmName, teleopPointer));
        mTeleopsPSMByPSM.insert(std::make_pair(psmName, teleopPointer));

        // first MTM with multiple PSMs is selected for single tap
        if ((mTeleopsPSMByMTM.count(mtmName) > 1)
            && (mTeleopMTMToCycle == "")) {
            mTeleopMTMToCycle = mtmName;
        }
        // check if we already have a teleop for the same PSM
        std::string mtmUsingThatPSM;
        GetMTMSelectedForPSM(psmName, mtmUsingThatPSM);
        if (mtmUsingThatPSM != "") {
            teleopPointer->SetSelected(false);
            CMN_LOG_CLASS_INIT_WARNING << "ConfigurePSMTeleopJSON: psm \""
                                       << psmName << "\" is already selected to be controlled by mtm \""
                                       << mtmUsingThatPSM << "\", component \""
                                       << name << "\" is added but not selected"
                                       << std::endl;
        } else {
            // check if we already have a teleop for the same PSM
            std::string psmUsingThatMTM;
            GetPSMSelectedForMTM(mtmName, psmUsingThatMTM);
            if (psmUsingThatMTM != "") {
                teleopPointer->SetSelected(false);
                CMN_LOG_CLASS_INIT_WARNING << "ConfigurePSMTeleopJSON: mtm \""
                                           << mtmName << "\" is already selected to control psm \""
                                           << psmUsingThatMTM << "\", component \""
                                           << name << "\" is added but not selected"
                                           << std::endl;
            } else {
                // neither the MTM nor PSM are used, let's activate that pair
                teleopPointer->SetSelected(true);
            }
        }
        // finally add the new teleop
        mTeleopsPSM[name] = teleopPointer;
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigurePSMTeleopJSON: there is already a teleop for the pair \""
                                 << name << "\"" << std::endl;
        return false;
    }

    jsonValue = jsonTeleop["type"];
    if (!jsonValue.empty()) {
        std::string typeString = jsonValue.asString();
        if (typeString == "TELEOP_PSM") {
            teleopPointer->m_type = TeleopPSM::TELEOP_PSM;
        } else if (typeString == "TELEOP_PSM_DERIVED") {
            teleopPointer->m_type = TeleopPSM::TELEOP_PSM_DERIVED;
        } else if (typeString == "TELEOP_PSM_GENERIC") {
            teleopPointer->m_type = TeleopPSM::TELEOP_PSM_GENERIC;
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "ConfigurePSMTeleopJSON: teleop " << name << ": invalid type \""
                                     << typeString << "\", needs to be TELEOP_PSM, TELEOP_PSM_DERIVED or TELEOP_PSM_GENERIC" << std::endl;
            return false;
        }
    } else {
        // default value
        teleopPointer->m_type = TeleopPSM::TELEOP_PSM;
    }

    // read period if present
    double period = mtsIntuitiveResearchKit::TeleopPeriod;
    jsonValue = jsonTeleop["period"];
    if (!jsonValue.empty()) {
        period = jsonValue.asFloat();
    }
    // for backward compatibility, send warning
    jsonValue = jsonTeleop["rotation"];
    if (!jsonValue.empty()) {
        CMN_LOG_CLASS_INIT_ERROR << "ConfigurePSMTeleopJSON: teleop " << name << ": \"rotation\" must now be defined under \"configure-parameter\" or in a separate configuration file" << std::endl;
        return false;
    }
    const Json::Value jsonTeleopConfig = jsonTeleop["configure-parameter"];
    teleopPointer->ConfigureTeleop(teleopPointer->m_type, period, jsonTeleopConfig);
    AddTeleopPSMInterfaces(teleopPointer);
    return true;
}

bool mtsIntuitiveResearchKitConsole::AddArmInterfaces(Arm * arm)
{
    // IO
    if (arm->expects_IO()) {
        const std::string interfaceNameIO = "IO-" + arm->Name();
        arm->IOInterfaceRequired = AddInterfaceRequired(interfaceNameIO);
        if (arm->IOInterfaceRequired) {
            arm->IOInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::ErrorEventHandler,
                                                           this, "error");
            arm->IOInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::WarningEventHandler,
                                                           this, "warning");
            arm->IOInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::StatusEventHandler,
                                                           this, "status");
            mConnections.Add(this->GetName(), interfaceNameIO,
                             arm->IOComponentName(), arm->Name());
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "AddArmInterfaces: failed to add IO interface for arm \""
                                     << arm->Name() << "\"" << std::endl;
            return false;
        }
        // is the arm is a PSM, since it has an IO, it also has a
        // Dallas chip interface and we want to see the messages
        if ((arm->m_type == Arm::ARM_PSM)
            || (arm->m_type == Arm::ARM_PSM_DERIVED)) {
            const std::string interfaceNameIODallas = "IO-Dallas-" + arm->Name();
            arm->IODallasInterfaceRequired = AddInterfaceRequired(interfaceNameIODallas);
            if (arm->IODallasInterfaceRequired) {
                arm->IODallasInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::ErrorEventHandler,
                                                                     this, "error");
                arm->IODallasInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::WarningEventHandler,
                                                                     this, "warning");
                arm->IODallasInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::StatusEventHandler,
                                                                     this, "status");
                mConnections.Add(this->GetName(), interfaceNameIODallas,
                                 arm->IOComponentName(), arm->Name() + "-Dallas");
            } else {
                CMN_LOG_CLASS_INIT_ERROR << "AddArmInterfaces: failed to add IO Dallas interface for arm \""
                                         << arm->Name() << "\"" << std::endl;
                return false;
            }
        }
    }

    // PID
    if (arm->expects_PID()) {
        const std::string interfaceNamePID = "PID-" + arm->Name();
        arm->PIDInterfaceRequired = AddInterfaceRequired(interfaceNamePID);
        if (arm->PIDInterfaceRequired) {
            arm->PIDInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::ErrorEventHandler,
                                                            this, "error");
            arm->PIDInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::WarningEventHandler,
                                                            this, "warning");
            arm->PIDInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::StatusEventHandler,
                                                            this, "status");
            mConnections.Add(this->GetName(), interfaceNamePID,
                             arm->Name() + "-PID", "Controller");
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "AddArmInterfaces: failed to add PID interface for arm \""
                                     << arm->Name() << "\"" << std::endl;
            return false;
        }
    }

    // arm interface
    const std::string interfaceNameArm = arm->Name();
    arm->ArmInterfaceRequired = AddInterfaceRequired(interfaceNameArm);
    if (arm->ArmInterfaceRequired) {
        arm->ArmInterfaceRequired->AddFunction("state_command", arm->state_command);
        if (!arm->suj()) {
            arm->ArmInterfaceRequired->AddFunction("hold", arm->hold, MTS_OPTIONAL);
        }
        arm->ArmInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::ErrorEventHandler,
                                                        this, "error");
        arm->ArmInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::WarningEventHandler,
                                                        this, "warning");
        arm->ArmInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::StatusEventHandler,
                                                        this, "status");
        arm->ArmInterfaceRequired->AddEventHandlerWrite(&mtsIntuitiveResearchKitConsole::Arm::CurrentStateEventHandler,
                                                        arm, "operating_state");
        mConnections.Add(this->GetName(), interfaceNameArm,
                         arm->ComponentName(), arm->InterfaceName());
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddArmInterfaces: failed to add Main interface for arm \""
                                 << arm->Name() << "\"" << std::endl;
        return false;
    }
    return true;
}

bool mtsIntuitiveResearchKitConsole::Connect(void)
{
    mConnections.Connect();

    mtsManagerLocal * componentManager = mtsManagerLocal::GetInstance();

    // connect console for audio feedback
    componentManager->Connect(this->GetName(), "TextToSpeech",
                              mTextToSpeech->GetName(), "Commands");

    for (auto & armIter : mArms) {
        Arm * arm = armIter.second;
        // arm specific interfaces
        arm->Connect();
        // connect to SUJ if needed
        if (arm->SUJInterfaceRequiredFromIO) {
            componentManager->Connect(this->GetName(), arm->SUJInterfaceRequiredFromIO->GetName(),
                                      arm->IOComponentName(), arm->Name() + "-SUJClutch");
        }
        if (arm->SUJInterfaceRequiredFromIO2) {
            componentManager->Connect(this->GetName(), arm->SUJInterfaceRequiredFromIO2->GetName(),
                                      arm->IOComponentName(), arm->Name() + "-SUJClutch2");
        }
        if (arm->SUJInterfaceRequiredToSUJ) {
            componentManager->Connect(this->GetName(), arm->SUJInterfaceRequiredToSUJ->GetName(),
                                      "SUJ", arm->Name());
        }
    }

    return true;
}

std::string mtsIntuitiveResearchKitConsole::locate_file(const std::string & filename)
{
    return m_config_path.Find(filename);
}

void mtsIntuitiveResearchKitConsole::power_off(void)
{
    teleop_enable(false);
    for (auto & arm : mArms) {
        arm.second->state_command(std::string("disable"));
    }
}

void mtsIntuitiveResearchKitConsole::power_on(void)
{
    DisableFaultyArms();
    for (auto & arm : mArms) {
        arm.second->state_command(std::string("enable"));
    }
}

void mtsIntuitiveResearchKitConsole::home(void)
{
    DisableFaultyArms();
    for (auto & arm : mArms) {
        arm.second->state_command(std::string("home"));
    }
}

void mtsIntuitiveResearchKitConsole::DisableFaultyArms(void)
{
    for (auto & arm : mArms) {
        auto armState = ArmStates.find(arm.first);
        if (// search if we already have a state
            (armState != ArmStates.end())
            // and the arm is faulty
            && (armState->second.State() == prmOperatingState::FAULT) ) {
            arm.second->state_command(std::string("disable"));
        }
    }
}

void mtsIntuitiveResearchKitConsole::teleop_enable(const bool & enable)
{
    mTeleopEnabled = enable;
    mTeleopDesired = enable;
    // event
    console_events.teleop_enabled(mTeleopEnabled);
    UpdateTeleopState();
}

void mtsIntuitiveResearchKitConsole::cycle_teleop_psm_by_mtm(const std::string & mtmName)
{
    // try to cycle through all the teleopPSMs associated to the MTM
    if (mTeleopsPSMByMTM.count(mtmName) == 0) {
        // we use empty string to query, no need to send warning about bad mtm name
        if (mtmName != "") {
            mInterface->SendWarning(this->GetName()
                                    + ": no PSM teleoperation found for MTM \""
                                    + mtmName
                                    + "\"");
        }
    } else if (mTeleopsPSMByMTM.count(mtmName) == 1) {
        mInterface->SendStatus(this->GetName()
                               + ": only one PSM teleoperation found for MTM \""
                               + mtmName
                               + "\", cycling has no effect");
    } else {
        // find range of teleops
        auto range = mTeleopsPSMByMTM.equal_range(mtmName);
        for (auto iter = range.first;
             iter != range.second;
             ++iter) {
            // find first teleop currently selected
            if (iter->second->Selected()) {
                // toggle to next one
                auto nextTeleop = iter;
                nextTeleop++;
                // if next one is last one, go back to first
                if (nextTeleop == range.second) {
                    nextTeleop = range.first;
                }
                // now make sure the PSM in next teleop is not used
                std::string mtmUsingThatPSM;
                GetMTMSelectedForPSM(nextTeleop->second->mPSMName, mtmUsingThatPSM);
                if (mtmUsingThatPSM != "") {
                    // message
                    mInterface->SendWarning(this->GetName()
                                            + ": cycling from \""
                                            + iter->second->m_name
                                            + "\" to \""
                                            + nextTeleop->second->m_name
                                            + "\" failed, PSM is already controlled by \""
                                            + mtmUsingThatPSM
                                            + "\"");
                } else {
                    // mark which one should be active
                    iter->second->SetSelected(false);
                    nextTeleop->second->SetSelected(true);
                    // if teleop PSM is active, enable/disable components now
                    if (mTeleopEnabled) {
                        iter->second->state_command(std::string("disable"));
                        if (mTeleopPSMRunning) {
                            nextTeleop->second->state_command(std::string("enable"));
                        } else {
                            nextTeleop->second->state_command(std::string("align_mtm"));
                        }
                    }
                    // message
                    mInterface->SendStatus(this->GetName()
                                           + ": cycling from \""
                                           + iter->second->m_name
                                           + "\" to \""
                                           + nextTeleop->second->m_name
                                           + "\"");
                }
                // stop for loop
                break;
            }
        }
    }
    // in all cases, emit events so users can figure out which components are selected
    EventSelectedTeleopPSMs();
}

void mtsIntuitiveResearchKitConsole::select_teleop_psm(const prmKeyValue & mtmPsm)
{
    // for readability
    const std::string mtmName = mtmPsm.Key;
    const std::string psmName = mtmPsm.Value;

    // if the psm value is empty, disable any teleop for the mtm -- this can be used to free the mtm
    if (psmName == "") {
        auto range = mTeleopsPSMByMTM.equal_range(mtmName);
        for (auto iter = range.first;
             iter != range.second;
             ++iter) {
            // look for the teleop that was selected if any
            if (iter->second->Selected()) {
                iter->second->SetSelected(false);
                // if teleop PSM is active, enable/disable components now
                if (mTeleopEnabled) {
                    iter->second->state_command(std::string("disable"));
                }
                // message
                mInterface->SendWarning(this->GetName()
                                        + ": teleop \""
                                        + iter->second->Name()
                                        + "\" has been unselected ");
            }
        }
        EventSelectedTeleopPSMs();
        return;
    }

    // actual teleop to select
    std::string name = mtmName + "-" + psmName;
    const auto teleopIterator = mTeleopsPSM.find(name);
    if (teleopIterator == mTeleopsPSM.end()) {
        mInterface->SendWarning(this->GetName()
                                + ": unable to select \""
                                + name
                                + "\", this component doesn't exist");
        EventSelectedTeleopPSMs();
        return;
    }
    // there seems to be some redundant information here, let's use it for a safety check
    CMN_ASSERT(mtmName == teleopIterator->second->mMTMName);
    CMN_ASSERT(psmName == teleopIterator->second->mPSMName);
    // check that the PSM is available to be used
    std::string mtmUsingThatPSM;
    GetMTMSelectedForPSM(psmName, mtmUsingThatPSM);
    if (mtmUsingThatPSM != "") {
        mInterface->SendWarning(this->GetName()
                                + ": unable to select \""
                                + name
                                + "\", PSM is already controlled by \""
                                + mtmUsingThatPSM
                                + "\"");
        EventSelectedTeleopPSMs();
        return;
    }

    // make sure the teleop using that MTM is unselected
    select_teleop_psm(prmKeyValue(mtmName, ""));

    // now turn on the teleop
    teleopIterator->second->SetSelected(true);
    // if teleop PSM is active, enable/disable components now
    if (mTeleopEnabled) {
        if (mTeleopPSMRunning) {
            teleopIterator->second->state_command(std::string("enable"));
        } else {
            teleopIterator->second->state_command(std::string("align_mtm"));
        }
    }
    // message
    mInterface->SendStatus(this->GetName()
                           + ": \""
                           + teleopIterator->second->m_name
                           + "\" has been selected");

    // always send a message to let user know the current status
    EventSelectedTeleopPSMs();
}

bool mtsIntuitiveResearchKitConsole::GetPSMSelectedForMTM(const std::string & mtmName, std::string & psmName) const
{
    bool mtmFound = false;
    psmName = "";
    // find range of teleops
    auto range = mTeleopsPSMByMTM.equal_range(mtmName);
    for (auto iter = range.first;
         iter != range.second;
         ++iter) {
        mtmFound = true;
        if (iter->second->Selected()) {
            psmName = iter->second->mPSMName;
        }
    }
    return mtmFound;
}

bool mtsIntuitiveResearchKitConsole::GetMTMSelectedForPSM(const std::string & psmName, std::string & mtmName) const
{
    bool psmFound = false;
    mtmName = "";
    for (auto & iter : mTeleopsPSM) {
        if (iter.second->mPSMName == psmName) {
            psmFound = true;
            if (iter.second->Selected()) {
                mtmName = iter.second->mMTMName;
            }
        }
    }
    return psmFound;
}

void mtsIntuitiveResearchKitConsole::EventSelectedTeleopPSMs(void) const
{
    for (auto & iter : mTeleopsPSM) {
        if (iter.second->Selected()) {
            ConfigurationEvents.teleop_psm_selected(prmKeyValue(iter.second->mMTMName,
                                                                iter.second->mPSMName));
        } else {
            ConfigurationEvents.teleop_psm_unselected(prmKeyValue(iter.second->mMTMName,
                                                                  iter.second->mPSMName));
        }
    }
}

void mtsIntuitiveResearchKitConsole::UpdateTeleopState(void)
{
    // Check if teleop is enabled
    if (!mTeleopEnabled) {
        bool holdNeeded = false;
        for (auto & iterTeleopPSM : mTeleopsPSM) {
            iterTeleopPSM.second->state_command(std::string("disable"));
            if (mTeleopPSMRunning) {
                holdNeeded = true;
            }
            mTeleopPSMRunning = false;
        }

        if (mTeleopECM) {
            mTeleopECM->state_command(std::string("disable"));
            if (mTeleopECMRunning) {
                holdNeeded = true;
            }
            mTeleopECMRunning = false;
        }

        // hold arms if we stopped any teleop
        if (holdNeeded) {
            for (auto & iterArms : mArms) {
                if (((iterArms.second->m_type == Arm::ARM_MTM) ||
                     (iterArms.second->m_type == Arm::ARM_MTM_DERIVED) ||
                     (iterArms.second->m_type == Arm::ARM_MTM_GENERIC))
                    && iterArms.second->hold.IsValid()) {
                    iterArms.second->hold();
                }
            }
        }
        return;
    }

    // if none are running, hold
    if (!mTeleopECMRunning && !mTeleopPSMRunning) {
        for (auto & iterArms : mArms) {
            if (((iterArms.second->m_type == Arm::ARM_MTM) ||
                 (iterArms.second->m_type == Arm::ARM_MTM_DERIVED) ||
                 (iterArms.second->m_type == Arm::ARM_MTM_GENERIC))
                && iterArms.second->hold.IsValid()) {
                iterArms.second->hold();
            }
        }
    }

    // all fine
    bool readyForTeleop = mOperatorPresent;

    for (auto & iterArms : mArms) {
        if (iterArms.second->mSUJClutched) {
            readyForTeleop = false;
        }
    }

    // Check if operator is present
    if (!readyForTeleop) {
        // keep MTMs aligned
        for (auto & iterTeleopPSM : mTeleopsPSM) {
            if (iterTeleopPSM.second->Selected()) {
                iterTeleopPSM.second->state_command(std::string("align_mtm"));
            } else {
                iterTeleopPSM.second->state_command(std::string("disable"));
            }
        }
        mTeleopPSMRunning = false;

        // stop ECM if needed
        if (mTeleopECMRunning) {
            mTeleopECM->state_command(std::string("disable"));
            mTeleopECMRunning = false;
        }
        return;
    }

    // If camera is pressed for ECM Teleop or not
    if (mCameraPressed) {
        if (!mTeleopECMRunning) {
            // if PSM was running so we need to stop it
            if (mTeleopPSMRunning) {
                for (auto & iterTeleopPSM : mTeleopsPSM) {
                    iterTeleopPSM.second->state_command(std::string("disable"));
                }
                mTeleopPSMRunning = false;
            }
            // ECM wasn't running, let's start it
            if (mTeleopECM) {
                mTeleopECM->state_command(std::string("enable"));
                mTeleopECMRunning = true;
            }
        }
    } else {
        // we must teleop PSM
        if (!mTeleopPSMRunning) {
            // if ECM was running so we need to stop it
            if (mTeleopECMRunning) {
                mTeleopECM->state_command(std::string("disable"));
                mTeleopECMRunning = false;
            }
            // PSM wasn't running, let's start it
            for (auto & iterTeleopPSM : mTeleopsPSM) {
                if (iterTeleopPSM.second->Selected()) {
                    iterTeleopPSM.second->state_command(std::string("enable"));
                } else {
                    iterTeleopPSM.second->state_command(std::string("disable"));
                }
                mTeleopPSMRunning = true;
            }
        }
    }
}

void mtsIntuitiveResearchKitConsole::set_scale(const double & scale)
{
    for (auto & iterTeleopPSM : mTeleopsPSM) {
        iterTeleopPSM.second->set_scale(scale);
    }
    ConfigurationEvents.scale(scale);
}

void mtsIntuitiveResearchKitConsole::set_volume(const double & volume)
{
    if (volume > 1.0) {
        m_audio_volume = 1.0;
    } else if (volume < 0.0) {
        m_audio_volume = 0.0;
    } else {
        m_audio_volume = volume;
    }
    std::stringstream message;
    message << this->GetName() << ": volume set to " << static_cast<int>(volume * 100.0) << "%";
    mInterface->SendStatus(message.str());
    audio.volume(m_audio_volume);
}

void mtsIntuitiveResearchKitConsole::beep(const vctDoubleVec & values)
{
    const size_t size = values.size();
    if ((size == 0) || (size > 3)) {
        mInterface->SendError(this->GetName() + ": beep expect up to 3 values (duration, frequency, volume)");
        return;
    }
    vctDoubleVec result(3);
    result.Assign(0.3, 3000.0, m_audio_volume);
    result.Ref(size).Assign(values); // overwrite with data sent
    // check duration
    bool durationError = false;
    if (result[0] < 0.1) {
        result[0] = 0.1;
        durationError = true;
    } else if (result[0] > 60.0) {
        result[0] = 60.0;
        durationError = true;
    }
    if (durationError) {
        mInterface->SendWarning(this->GetName() + ": beep, duration must be between 0.1 and 60s");
    }
    // check volume
    bool volumeError = false;
    if (result[2] < 0.0) {
        result[2] = 0.0;
        volumeError = true;
    } else if (result[2] > 1.0) {
        result[2] = 1.0;
        volumeError = true;
    }
    if (volumeError) {
        mInterface->SendWarning(this->GetName() + ": beep, volume must be between 0 and 1");
    }
    // convert to fixed size vector and send
    audio.beep(vct3(result));
}

void mtsIntuitiveResearchKitConsole::string_to_speech(const std::string & text)
{
    audio.string_to_speech(text);
}

void mtsIntuitiveResearchKitConsole::ClutchEventHandler(const prmEventButton & button)
{
    switch (button.Type()) {
    case prmEventButton::PRESSED:
        mInterface->SendStatus(this->GetName() + ": clutch pressed");
        audio.beep(vct3(0.1, 700.0, m_audio_volume));
        break;
    case prmEventButton::RELEASED:
        mInterface->SendStatus(this->GetName() + ": clutch released");
        audio.beep(vct3(0.1, 700.0, m_audio_volume));
        break;
    case prmEventButton::CLICKED:
        mInterface->SendStatus(this->GetName() + ": clutch quick tap");
        audio.beep(vct3(0.05, 2000.0, m_audio_volume));
        audio.beep(vct3(0.05, 2000.0, m_audio_volume));
        if (mTeleopMTMToCycle != "") {
            cycle_teleop_psm_by_mtm(mTeleopMTMToCycle);
        }
        break;
    default:
        break;
    }
    console_events.clutch(button);
}

void mtsIntuitiveResearchKitConsole::CameraEventHandler(const prmEventButton & button)
{
    switch (button.Type()) {
    case prmEventButton::PRESSED:
        mCameraPressed = true;
        mInterface->SendStatus(this->GetName() + ": camera pressed");
        audio.beep(vct3(0.1, 1000.0, m_audio_volume));
        break;
    case prmEventButton::RELEASED:
        mCameraPressed = false;
        mInterface->SendStatus(this->GetName() + ": camera released");
        audio.beep(vct3(0.1, 1000.0, m_audio_volume));
        break;
    case prmEventButton::CLICKED:
        mInterface->SendStatus(this->GetName() + ": camera quick tap");
        audio.beep(vct3(0.05, 2500.0, m_audio_volume));
        audio.beep(vct3(0.05, 2500.0, m_audio_volume));
        break;
    default:
        break;
    }
    UpdateTeleopState();
    console_events.camera(button);
}

void mtsIntuitiveResearchKitConsole::OperatorPresentEventHandler(const prmEventButton & button)
{
    switch (button.Type()) {
    case prmEventButton::PRESSED:
        mOperatorPresent = true;
        mInterface->SendStatus(this->GetName() + ": operator present");
        audio.beep(vct3(0.3, 1500.0, m_audio_volume));
        break;
    case prmEventButton::RELEASED:
        mOperatorPresent = false;
        mInterface->SendStatus(this->GetName() + ": operator not present");
        audio.beep(vct3(0.3, 1200.0, m_audio_volume));
        break;
    default:
        break;
    }
    UpdateTeleopState();
    console_events.operator_present(button);
}

void mtsIntuitiveResearchKitConsole::TeleopScaleChangedEventHandler(const double & scale)
{
    ConfigurationEvents.scale(scale);
}

void mtsIntuitiveResearchKitConsole::ErrorEventHandler(const mtsMessage & message)
{
    // similar to teleop_enable(false) except we don't change mTeleopDesired
    mTeleopEnabled = false;
    console_events.teleop_enabled(mTeleopEnabled);
    UpdateTeleopState();

    mInterface->SendError(message.Message);
    // throttle error beeps
    double currentTime = mtsManagerLocal::GetInstance()->GetTimeServer().GetRelativeTime();
    if ((currentTime - mTimeOfLastErrorBeep) > 2.0 * cmn_s) {
        audio.beep(vct3(0.3, 3000.0, m_audio_volume));
        mTimeOfLastErrorBeep = currentTime;
    }
}

void mtsIntuitiveResearchKitConsole::WarningEventHandler(const mtsMessage & message)
{
    mInterface->SendWarning(message.Message);
}

void mtsIntuitiveResearchKitConsole::StatusEventHandler(const mtsMessage & message)
{
    mInterface->SendStatus(message.Message);
}

void mtsIntuitiveResearchKitConsole::SetArmCurrentState(const std::string & arm_name,
                                                        const prmOperatingState & currentState)
{
    if (mTeleopDesired) {
        auto armState = ArmStates.find(arm_name);
        bool newArm = (armState == ArmStates.end());
        if (newArm) {
            teleop_enable(true);
        } else {
            // for all arms update the state if it was not enabled and the new
            // state is enabled, home and not busy
            if (((armState->second.State() != prmOperatingState::ENABLED)
                 && currentState.IsEnabledHomedAndNotBusy())) {
                teleop_enable(true);
            } else {
                // special case for PSMs when coming back from busy state
                // (e.g. engaging adapter or instrument)
                const auto count = mTeleopsPSMByPSM.count(arm_name);
                if (count != 0) {
                    if (!armState->second.IsBusy() && currentState.IsEnabledHomedAndNotBusy()) {
                        teleop_enable(true);
                    }
                }
            }
        }
    }

    // save state
    ArmStates[arm_name] = currentState;

    // emit event (for Qt GUI)
    std::string payload = "";
    if (currentState.IsEnabledAndHomed()) {
        payload = "ENABLED";
    } else if (currentState.State() == prmOperatingState::FAULT) {
        payload = "FAULT";
    }
    ConfigurationEvents.ArmCurrentState(prmKeyValue(arm_name, payload));
}
