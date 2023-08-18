/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  Author(s):  Zihan Chen, Anton Deguet
  Created on: 2013-02-20

  (C) Copyright 2013-2023 Johns Hopkins University (JHU), All Rights Reserved.

  --- begin cisst license - do not edit ---

  This software is provided "as is" under an open source license, with
  no warranty.  The complete license can be found in license.txt and
  http://www.cisst.org/cisst/license.txt.

  --- end cisst license ---
*/

// system include
#include <iostream>
#include <sstream>
#include <fstream>

// cisst
#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKit.h>
#include <sawIntuitiveResearchKit/mtsTeleOperationPSM.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisstParameterTypes/prmOperatingState.h>

CMN_IMPLEMENT_SERVICES_DERIVED_ONEARG(mtsTeleOperationPSM, mtsTaskPeriodic, mtsTaskPeriodicConstructorArg);

mtsTeleOperationPSM::mtsTeleOperationPSM(const std::string & componentName, const double periodInSeconds):
    mtsTaskPeriodic(componentName, periodInSeconds),
    mTeleopMode(Mode::UNILATERAL),
    mTeleopState(componentName, "DISABLED")
{
    Init();
}

mtsTeleOperationPSM::mtsTeleOperationPSM(const mtsTaskPeriodicConstructorArg & arg):
    mtsTaskPeriodic(arg),
    mTeleopMode(Mode::UNILATERAL),
    mTeleopState(arg.Name, "DISABLED")
{
    Init();
}

mtsTeleOperationPSM::~mtsTeleOperationPSM()
{
}

mtsTeleOperationPSM::Result::Result(bool ok, std::string message, mtsExecutionResult executionResult):
    ok(ok)
{
    std::stringstream ss;
    ss << message << ": \"" << executionResult << "\"";
    this->message = ss.str();
}

void mtsTeleOperationPSM::Arm::populateInterface(mtsInterfaceRequired* interfaceRequired) {
    interfaceRequired->AddFunction("operating_state", operating_state);
    interfaceRequired->AddFunction("state_command", state_command);

    interfaceRequired->AddFunction("measured_js", measured_js);
    interfaceRequired->AddFunction("measured_cp", measured_cp);
    interfaceRequired->AddFunction("measured_cv", body_measured_cv);
    interfaceRequired->AddFunction("body/measured_cf", body_measured_cf);
    interfaceRequired->AddFunction("external/measured_cf", body_external_cf);
    interfaceRequired->AddFunction("setpoint_cp", setpoint_cp);
    interfaceRequired->AddFunction("setpoint_js", setpoint_js);
    interfaceRequired->AddFunction("servo_cpvf", servo_cpvf);
}

prmStateCartesian mtsTeleOperationPSM::Arm::computeGoalFromTarget(Arm* target, const vctMatRot3& alignment_offset, double size_scale) const
{
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // NOTE: we need take into account changes in PSM base frame if any
    // base frame affects all three of position, velocity, and effort
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    prmStateCartesian goalState;

    vctFrm4x4 position(m_measured_cp.Position());
    vctFrm4x4 targetPosition(target->m_measured_cp.Position());

    // translation
    vct3 relativeTranslation = targetPosition.Translation() - target->CartesianInitial.Translation();
    vct3 goalTranslation = CartesianInitial.Translation() + size_scale * relativeTranslation;

    // rotation
    vctMatRot3 goalRotation = vctMatRot3(targetPosition.Rotation() * alignment_offset);

    // desired Cartesian position
    vctFrm4x4 goalPosition;
    goalPosition.Translation().Assign(goalTranslation);
    goalPosition.Rotation().FromNormalized(goalRotation);

    goalState.Position().FromNormalized(goalPosition);
    goalState.PositionIsDefined() = true;

    auto goalAngVelocity = target->m_body_measured_cv.VelocityAngular();
    auto targetVelocity = target->m_body_measured_cv.VelocityLinear();

    // linear component is scaled and re-oriented
    goalState.Velocity().Ref<3>(0).Assign(size_scale * targetVelocity);
    // angular component is not scaled
    goalState.Velocity().Ref<3>(3).Assign(goalAngVelocity);
    goalState.VelocityIsDefined() = true;

    vct6 targetForce = target->m_body_measured_cf.Force();
    vct6 localForce = m_body_measured_cf.Force();

    vct6 sum = (-targetForce) - localForce;
    goalState.Effort().Assign(sum);
    goalState.EffortIsDefined() = true;

    return goalState;
}

prmStateCartesian mtsTeleOperationPSM::ArmMTM::computeGoalFromTarget(Arm* target, const vctMatRot3& alignment_offset, double size_scale) const
{
    auto goalState = Arm::computeGoalFromTarget(target, alignment_offset, size_scale);

    vct6 targetForce = target->m_body_external_cf.Force();
    vct6 localForce = m_body_external_cf.Force();

    vct6 force_sum = (-targetForce) - localForce;
    goalState.Effort().Assign(force_sum);
    goalState.EffortIsDefined() = true;

    return goalState;
}

prmStateCartesian mtsTeleOperationPSM::ArmPSM::computeGoalFromTarget(Arm* target, const vctMatRot3& alignment_offset, double size_scale) const
{
    auto goalState = Arm::computeGoalFromTarget(target, alignment_offset, size_scale);

    vct6 targetForce = target->m_body_external_cf.Force();
    vct6 localForce = m_body_external_cf.Force();

    vct6 force_sum = (-targetForce) - localForce;
    goalState.Effort().Assign(force_sum);
    goalState.EffortIsDefined() = true;

    return goalState;
}

void mtsTeleOperationPSM::ArmMTM::populateInterface(mtsInterfaceRequired* interfaceRequired) {
    Arm::populateInterface(interfaceRequired);

    interfaceRequired->AddFunction("move_cp", move_cp);
    interfaceRequired->AddFunction("gripper/measured_js", gripper_measured_js, MTS_OPTIONAL);
    interfaceRequired->AddFunction("lock_orientation", lock_orientation, MTS_OPTIONAL);
    interfaceRequired->AddFunction("unlock_orientation", unlock_orientation, MTS_OPTIONAL);
    interfaceRequired->AddFunction("body/servo_cf", body_servo_cf, MTS_OPTIONAL);
}

void mtsTeleOperationPSM::ArmPSM::populateInterface(mtsInterfaceRequired* interfaceRequired) {
    Arm::populateInterface(interfaceRequired);

    interfaceRequired->AddFunction("hold", hold);
    interfaceRequired->AddFunction("jaw/setpoint_js", jaw_setpoint_js, MTS_OPTIONAL);
    interfaceRequired->AddFunction("jaw/measured_js", jaw_measured_js, MTS_OPTIONAL);
    interfaceRequired->AddFunction("jaw/configuration_js", jaw_configuration_js, MTS_OPTIONAL);
    interfaceRequired->AddFunction("jaw/servo_jp", jaw_servo_jp, MTS_OPTIONAL);
}

mtsTeleOperationPSM::Result mtsTeleOperationPSM::Arm::getData() {
    mtsExecutionResult execution_result;

    execution_result = measured_cp(m_measured_cp);
    if (!execution_result.IsOK()) { return Result(false, "call to measured_cp failed", execution_result); }

    execution_result = body_measured_cv(m_body_measured_cv);
    if (!execution_result.IsOK()) { return Result(false, "call to measured_cv failed", execution_result); }

    execution_result = measured_js(m_measured_js);
    if (!execution_result.IsOK()) { return Result(false, "call to measured_js failed", execution_result); }

    execution_result = body_measured_cf(m_body_measured_cf);
    if (!execution_result.IsOK()) { return Result(false, "call to body_measured_cf failed", execution_result); }

    execution_result = body_external_cf(m_body_external_cf);
    if (!execution_result.IsOK()) { return Result(false, "call to body_external_cf failed", execution_result); }

    execution_result = setpoint_cp(m_setpoint_cp);
    if (!execution_result.IsOK()) { return Result(false, "call to setpoint_cp failed", execution_result); }

    execution_result = setpoint_js(m_setpoint_js);
    if (!execution_result.IsOK()) { return Result(false, "call to setpoint_js failed", execution_result); }

    if (sampleDelay > 0) {
        cp_delay_buffer.push_back(m_measured_cp);
        cv_delay_buffer.push_back(m_body_measured_cv);
        cf_delay_buffer.push_back(m_body_measured_cf);
        cf_external_delay_buffer.push_back(m_body_external_cf);

        // if someone *decreased* artifical latency while running
        while (cp_delay_buffer.size() > sampleDelay) {
            cp_delay_buffer.pop_front();
            cv_delay_buffer.pop_front();
            cf_delay_buffer.pop_front();
            cf_external_delay_buffer.pop_front();
        }

        double current_time = m_measured_cp.Timestamp();

        m_measured_cp = cp_delay_buffer.front();
        m_body_measured_cv = cv_delay_buffer.front();
        m_body_measured_cf = cf_delay_buffer.front();
        m_body_external_cf = cf_external_delay_buffer.front();

        double delayed_time = m_measured_cp.Timestamp();

        if (cp_delay_buffer.size() >= sampleDelay) {
            cp_delay_buffer.pop_front();
            cv_delay_buffer.pop_front();
            cf_delay_buffer.pop_front();
            cf_external_delay_buffer.pop_front();
        }
    }

    return Result(true, "", execution_result);
}

mtsTeleOperationPSM::Result mtsTeleOperationPSM::ArmMTM::getData() {
    if (gripper_measured_js.IsValid()) {
        mtsExecutionResult execution_result = gripper_measured_js(m_gripper_measured_js);
        if (!execution_result.IsOK()) { return Result(false, "call to gripper_measured_js failed", execution_result); }

        if (sampleDelay > 0) {
            gripper_delay_buffer.push_back(m_gripper_measured_js);

            // if someone *decreased* artifical latency while running
            while (gripper_delay_buffer.size() > sampleDelay) {
                gripper_delay_buffer.pop_front();
            }

            m_gripper_measured_js = gripper_delay_buffer.front();

            if (gripper_delay_buffer.size() >= sampleDelay) {
                gripper_delay_buffer.pop_front();
            }
        }
    }

    return Arm::getData();
}

mtsTeleOperationPSM::Result mtsTeleOperationPSM::ArmPSM::getData() {
    return Arm::getData();
}

void mtsTeleOperationPSM::Init(void)
{
    // configure state machine
    mTeleopState.AddState("SETTING_ARMS_STATE");
    mTeleopState.AddState("ALIGNING_MTM");
    mTeleopState.AddState("ENABLED");
    mTeleopState.AddAllowedDesiredState("ENABLED");
    mTeleopState.AddAllowedDesiredState("ALIGNING_MTM");
    mTeleopState.AddAllowedDesiredState("DISABLED");

    // state change, to convert to string events for users (Qt, ROS)
    mTeleopState.SetStateChangedCallback(&mtsTeleOperationPSM::StateChanged,
                                         this);

    // run for all states
    mTeleopState.SetRunCallback(&mtsTeleOperationPSM::RunAllStates,
                                this);

    // disabled
    mTeleopState.SetTransitionCallback("DISABLED",
                                       &mtsTeleOperationPSM::TransitionDisabled,
                                       this);

    // setting arms state
    mTeleopState.SetEnterCallback("SETTING_ARMS_STATE",
                                  &mtsTeleOperationPSM::EnterSettingArmsState,
                                  this);
    mTeleopState.SetTransitionCallback("SETTING_ARMS_STATE",
                                       &mtsTeleOperationPSM::TransitionSettingArmsState,
                                       this);

    // aligning MTM
    mTeleopState.SetEnterCallback("ALIGNING_MTM",
                                  &mtsTeleOperationPSM::EnterAligningMTM,
                                  this);
    mTeleopState.SetRunCallback("ALIGNING_MTM",
                                &mtsTeleOperationPSM::RunAligningMTM,
                                this);
    mTeleopState.SetTransitionCallback("ALIGNING_MTM",
                                       &mtsTeleOperationPSM::TransitionAligningMTM,
                                       this);

    // enabled
    mTeleopState.SetEnterCallback("ENABLED",
                                  &mtsTeleOperationPSM::EnterEnabled,
                                  this);
    mTeleopState.SetRunCallback("ENABLED",
                                &mtsTeleOperationPSM::RunEnabled,
                                this);
    mTeleopState.SetTransitionCallback("ENABLED",
                                       &mtsTeleOperationPSM::TransitionEnabled,
                                       this);

    mPSM.m_jaw_servo_jp.Goal().SetSize(1);

    this->StateTable.AddData(mMTM.m_measured_cp, "MTM/measured_cp");
    this->StateTable.AddData(mMTM.m_initial_cp, "MTM/initial_cp");
    this->StateTable.AddData(mMTM.m_body_measured_cv, "MTM/measured_cv");
    this->StateTable.AddData(mMTM.m_body_measured_cf, "MTM/body_measured_cf");
    this->StateTable.AddData(mMTM.m_setpoint_cp, "MTM/setpoint_cp");
    this->StateTable.AddData(mMTM.m_setpoint_cp, "MTM/setpoint_cp");
    this->StateTable.AddData(mPSM.m_measured_cp, "PSM/measured_cp");
    this->StateTable.AddData(mPSM.m_initial_cp, "PSM/initial_cp");
    this->StateTable.AddData(mPSM.m_body_measured_cv, "PSM/measured_cv");
    this->StateTable.AddData(mPSM.m_body_measured_cf, "PSM/body_measured_cf");
    this->StateTable.AddData(mPSM.m_setpoint_cp, "PSM/setpoint_cp");
    this->StateTable.AddData(m_alignment_offset, "alignment_offset");

    mConfigurationStateTable = new mtsStateTable(100, "Configuration");
    mConfigurationStateTable->SetAutomaticAdvance(false);
    this->AddStateTable(mConfigurationStateTable);
    mConfigurationStateTable->AddData(m_scale, "scale");
    mConfigurationStateTable->AddData(m_rotation_locked, "rotation_locked");
    mConfigurationStateTable->AddData(m_translation_locked, "translation_locked");
    mConfigurationStateTable->AddData(m_align_mtm, "align_mtm");

    // setup cisst interfaces
    mtsInterfaceRequired * interfaceRequired = AddInterfaceRequired("MTM");
    if (interfaceRequired) {
        mMTM.populateInterface(interfaceRequired);
        interfaceRequired->AddEventHandlerWrite(&mtsTeleOperationPSM::MTMErrorEventHandler, 
                                                this, "error");
    }

    interfaceRequired = AddInterfaceRequired("PSM");
    if (interfaceRequired) {
        mPSM.populateInterface(interfaceRequired);
        interfaceRequired->AddEventHandlerWrite(&mtsTeleOperationPSM::PSMErrorEventHandler,
                                                this, "error");
    }

    // footpedal events
    interfaceRequired = AddInterfaceRequired("Clutch");
    if (interfaceRequired) {
        interfaceRequired->AddEventHandlerWrite(&mtsTeleOperationPSM::ClutchEventHandler, this, "Button");
    }

    interfaceRequired = AddInterfaceRequired("PSM-base-frame", MTS_OPTIONAL);
    if (interfaceRequired) {
        interfaceRequired->AddFunction("measured_cp", mBaseFrame.measured_cp);
    }

    mInterface = AddInterfaceProvided("Setting");
    if (mInterface) {
        mInterface->AddMessageEvents();
        // commands
        mInterface->AddCommandReadState(StateTable, StateTable.PeriodStats,
                                        "period_statistics"); // mtsIntervalStatistics

        mInterface->AddCommandWrite(&mtsTeleOperationPSM::state_command, this,
                                    "state_command", std::string());
        mInterface->AddCommandWrite(&mtsTeleOperationPSM::set_scale, this,
                                    "set_scale", m_scale);
        mInterface->AddCommandWrite(&mtsTeleOperationPSM::lock_rotation, this,
                                    "lock_rotation", m_rotation_locked);
        mInterface->AddCommandWrite(&mtsTeleOperationPSM::lock_translation, this,
                                    "lock_translation", m_translation_locked);
        mInterface->AddCommandWrite(&mtsTeleOperationPSM::set_align_mtm, this,
                                    "set_align_mtm", m_align_mtm);
        mInterface->AddCommandReadState(*(mConfigurationStateTable),
                                        m_scale,
                                        "scale");
        mInterface->AddCommandReadState(*(mConfigurationStateTable),
                                        m_rotation_locked, "rotation_locked");
        mInterface->AddCommandReadState(*(mConfigurationStateTable),
                                        m_translation_locked, "translation_locked");
        mInterface->AddCommandReadState(*(mConfigurationStateTable),
                                        m_align_mtm, "align_mtm");
        mInterface->AddCommandReadState(this->StateTable,
                                        mMTM.m_measured_cp,
                                        "MTM/measured_cp");
        mInterface->AddCommandReadState(this->StateTable,
                                        mMTM.m_body_measured_cv,
                                        "MTM/measured_cv");
        mInterface->AddCommandReadState(this->StateTable,
                                        mPSM.m_setpoint_cp,
                                        "PSM/setpoint_cp");
        mInterface->AddCommandReadState(this->StateTable,
                                        m_alignment_offset,
                                        "alignment_offset");

        mInterface->AddCommandReadState(this->StateTable,
                                        mMTM.m_initial_cp,
                                        "MTM/initial/measured_cp");

        mInterface->AddCommandReadState(this->StateTable,
                                        mPSM.m_initial_cp,
                                        "PSM/initial/measured_cp");
        // events
        mInterface->AddEventWrite(MessageEvents.desired_state,
                                  "desired_state", std::string(""));
        mInterface->AddEventWrite(MessageEvents.current_state,
                                  "current_state", std::string(""));
        mInterface->AddEventWrite(MessageEvents.following,
                                  "following", false);
        // configuration
        mInterface->AddEventWrite(ConfigurationEvents.scale,
                                  "scale", m_scale);
        mInterface->AddEventWrite(ConfigurationEvents.rotation_locked,
                                  "rotation_locked", m_rotation_locked);
        mInterface->AddEventWrite(ConfigurationEvents.translation_locked,
                                  "translation_locked", m_translation_locked);
        mInterface->AddEventWrite(ConfigurationEvents.align_mtm,
                                  "align_mtm", m_align_mtm);
    }

    // so sent commands can be used with ros-bridge
    mPSM.m_servo_cpvf.Valid() = true;
    mPSM.m_jaw_servo_jp.Valid() = true;
}

void mtsTeleOperationPSM::Configure(const std::string & filename)
{
    std::ifstream jsonStream;
    Json::Value jsonConfig;
    Json::Reader jsonReader;

    if (filename == "") {
        return;
    }

    jsonStream.open(filename.c_str());
    if (!jsonReader.parse(jsonStream, jsonConfig)) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                 << ": failed to parse configuration file \""
                                 << filename << "\"\n"
                                 << jsonReader.getFormattedErrorMessages();
        exit(EXIT_FAILURE);
    }

    CMN_LOG_CLASS_INIT_VERBOSE << "Configure: " << this->GetName()
                               << " using file \"" << filename << "\"" << std::endl
                               << "----> content of configuration file: " << std::endl
                               << jsonConfig << std::endl
                               << "<----" << std::endl;

    // base component configuration
    mtsComponent::ConfigureJSON(jsonConfig);

    // JSON part
    mtsTeleOperationPSM::Configure(jsonConfig);
}

void mtsTeleOperationPSM::Configure(const Json::Value & jsonConfig)
{
    Json::Value jsonValue;

    // base component configuration
    mtsComponent::ConfigureJSON(jsonConfig);

    // read teleop mode if present
    jsonValue = jsonConfig["mode"];
    if (!jsonValue.empty()) {
        const auto teleopMode = jsonValue.asString();
        if (teleopMode == "UNILATERAL") {
            mTeleopMode = Mode::UNILATERAL;
        } else if (teleopMode == "BILATERAL") {
            mTeleopMode = Mode::BILATERAL;
        } else if (teleopMode == "HIGH_LATENCY") {
            mTeleopMode = Mode::HIGH_LATENCY;
        } else {
            const std::string options = "UNILATERAL, BILATERAL, HIGH_LATENCY";
            CMN_LOG_CLASS_INIT_ERROR << "Configure: " << this->GetName()
                                     << " mode \"" << teleopMode << "\" is not valid.  Valid options are: "
                                     << options << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // read scale if present
    jsonValue = jsonConfig["scale"];
    if (!jsonValue.empty()) {
        m_scale = jsonValue.asDouble();
    }
    if (m_scale <= 0.0) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                 << ": \"scale\" must be a positive number.  Found " << m_scale << std::endl;
        exit(EXIT_FAILURE);
    }

    // read artificial-latency if present
    jsonValue = jsonConfig["artificial-latency-ms"];
    if (!jsonValue.empty()) {
        double latency = jsonValue.asDouble();
        if (latency < 0.0) {
            CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                     << ": \"artificial-latency-ms\" must be a non-negative number.  Found " << latency << std::endl;
            exit(EXIT_FAILURE);
        }
        int sample_delay = (int)(latency * (0.001 / (GetPeriodicity() + 0.00006)));
        mPSM.sampleDelay = sample_delay;
        mMTM.sampleDelay = sample_delay;
    }

    // read orientation if present
    jsonValue = jsonConfig["rotation"];
    if (!jsonValue.empty()) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                 << ": \"rotation\" is deprecated"<< std::endl;
        exit(EXIT_FAILURE);
    }

    // rotation locked
    jsonValue = jsonConfig["rotation-locked"];
    if (!jsonValue.empty()) {
        m_rotation_locked = jsonValue.asBool();
    }

    // rotation locked
    jsonValue = jsonConfig["translation-locked"];
    if (!jsonValue.empty()) {
        m_translation_locked = jsonValue.asBool();
    }

    // ignore jaw if needed
    jsonValue = jsonConfig["ignore-jaw"];
    if (!jsonValue.empty()) {
        m_jaw.ignore = jsonValue.asBool();
    }

    // jaw rate of opening-closing
    jsonValue = jsonConfig["jaw-rate"];
    if (!jsonValue.empty()) {
        m_jaw.rate = jsonValue.asDouble();
    }
    if (m_jaw.rate <= 0.0) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                 << ": \"jaw-rate\" must be a positive number.  Found " << m_jaw.rate << std::endl;
        exit(EXIT_FAILURE);
    }

    // jaw rate of opening-closing after clutch
    jsonValue = jsonConfig["jaw-rate-back-from-clutch"];
    if (!jsonValue.empty()) {
        m_jaw.rate_back_from_clutch = jsonValue.asDouble();
    }
    if (m_jaw.rate_back_from_clutch <= 0.0) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                 << ": \"jaw-rate-back-from-clutch\" must be a positive number.  Found " << m_jaw.rate << std::endl;
        exit(EXIT_FAILURE);
    }

    // gripper scaling
    Json::Value jsonGripper = jsonConfig["gripper-scaling"];
    if (!jsonGripper.empty()) {
        jsonValue = jsonGripper["max"];
        if (!jsonValue.empty()) {
            m_gripper.max = jsonValue.asDouble();
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                     << ": \"gripper-scaling\": { \"max\": } is missing" << std::endl;
            exit(EXIT_FAILURE);
        }
        jsonValue = jsonGripper["zero"];
        if (!jsonValue.empty()) {
            m_gripper.zero = jsonValue.asDouble();
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                     << ": \"gripper-scaling\": { \"zero\": } is missing" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // orientation tolerance to start teleop
    jsonValue = jsonConfig["start-orientation-tolerance"];
    if (!jsonValue.empty()) {
        m_operator.orientation_tolerance = jsonValue.asDouble();
    }
    if (m_operator.orientation_tolerance < 0.0) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                 << ": \"start-orientation-tolerance\" must be a positive number.  Found "
                                 << m_operator.orientation_tolerance << std::endl;
        exit(EXIT_FAILURE);
    }

    // Gripper threshold to start teleop
    jsonValue = jsonConfig["start-gripper-threshold"];
    if (!jsonValue.empty()) {
        m_operator.gripper_threshold = jsonValue.asDouble();
    }
    if (m_operator.gripper_threshold < 0.0) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                 << ": \"start-gripper-threshold\" must be a positive number.  Found "
                                 << m_operator.gripper_threshold << std::endl;
        exit(EXIT_FAILURE);
    }

    // roll threshold to start teleop
    jsonValue = jsonConfig["start-roll-threshold"];
    if (!jsonValue.empty()) {
        m_operator.roll_threshold = jsonValue.asDouble();
    }
    if (m_operator.roll_threshold < 0.0) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure " << this->GetName()
                                 << ": \"start-roll-threshold\" must be a positive number.  Found "
                                 << m_operator.roll_threshold << std::endl;
        exit(EXIT_FAILURE);
    }

    // align MTM if needed
    jsonValue = jsonConfig["align-mtm"];
    if (!jsonValue.empty()) {
        m_align_mtm = jsonValue.asBool();
    }
}

void mtsTeleOperationPSM::Startup(void)
{
    CMN_LOG_CLASS_INIT_VERBOSE << "Startup" << std::endl;
    set_scale(m_scale);
    set_following(false);
    lock_rotation(m_rotation_locked);
    lock_translation(m_translation_locked);
    set_align_mtm(m_align_mtm);

    // check if functions for jaw are connected
    if (!m_jaw.ignore) {
        if (!mPSM.jaw_setpoint_js.IsValid()
            || !mPSM.jaw_servo_jp.IsValid()) {
            mInterface->SendError(this->GetName() + ": optional functions \"jaw/servo_jp\" and \"jaw/setpoint_js\" are not connected, setting \"ignore-jaw\" to true");
            m_jaw.ignore = true;
        }
    }
}

void mtsTeleOperationPSM::Run(void)
{
    ProcessQueuedCommands();
    ProcessQueuedEvents();

    // run based on state
    mTeleopState.Run();
}

void mtsTeleOperationPSM::Cleanup(void)
{
    CMN_LOG_CLASS_INIT_VERBOSE << "Cleanup" << std::endl;
}

void mtsTeleOperationPSM::MTMErrorEventHandler(const mtsMessage & message)
{
    mTeleopState.SetDesiredState("DISABLED");
    mInterface->SendError(this->GetName() + ": received from MTM [" + message.Message + "]");
}

void mtsTeleOperationPSM::PSMErrorEventHandler(const mtsMessage & message)
{
    mTeleopState.SetDesiredState("DISABLED");
    mInterface->SendError(this->GetName() + ": received from PSM [" + message.Message + "]");
}

void mtsTeleOperationPSM::ClutchEventHandler(const prmEventButton & button)
{
    switch (button.Type()) {
    case prmEventButton::PRESSED:
        m_clutched = true;
        break;
    case prmEventButton::RELEASED:
        m_clutched = false;
        break;
    default:
        break;
    }

    // if the teleoperation is activated
    if (mTeleopState.DesiredState() == "ENABLED") {
        Clutch(m_clutched);
    }
}

void mtsTeleOperationPSM::Clutch(const bool & clutch)
{
    // if the teleoperation is activated
    if (clutch) {
        // keep track of last follow mode
        m_operator.was_active_before_clutch = m_operator.is_active;
        set_following(false);
        mMTM.m_move_cp.Goal().Rotation().FromNormalized(mPSM.m_setpoint_cp.Position().Rotation());
        mMTM.m_move_cp.Goal().Translation().Assign(mMTM.m_measured_cp.Position().Translation());
        mInterface->SendStatus(this->GetName() + ": console clutch pressed");

        if ((m_align_mtm || m_rotation_locked)
            && mMTM.lock_orientation.IsValid()) {
            // lock in current position
            mMTM.lock_orientation(mMTM.m_measured_cp.Position().Rotation());
        } else {
            // make sure it is freed
            if (mMTM.unlock_orientation.IsValid()) {
                mMTM.unlock_orientation();
            }
        }

        // make sure PSM stops moving
        mPSM.hold();
    } else {
        mInterface->SendStatus(this->GetName() + ": console clutch released");
        mTeleopState.SetCurrentState("SETTING_ARMS_STATE");
        m_back_from_clutch = true;
        m_jaw_caught_up_after_clutch = false;
    }
}


void mtsTeleOperationPSM::state_command(const std::string & command)
{
    if (command == "enable") {
        SetDesiredState("ENABLED");
        return;
    }
    if (command == "disable") {
        SetDesiredState("DISABLED");
        return;
    }
    if (command == "align_mtm") {
        SetDesiredState("ALIGNING_MTM");
        return;
    }
    mInterface->SendWarning(this->GetName() + ": " + command + " doesn't seem to be a valid state_command");
}


void mtsTeleOperationPSM::SetDesiredState(const std::string & state)
{
    // try to find the state in state machine
    if (!mTeleopState.StateExists(state)) {
        mInterface->SendError(this->GetName() + ": unsupported state " + state);
        return;
    }
    // return is already the desired state
    if (mTeleopState.DesiredState() == state) {
        MessageEvents.desired_state(state);
        return;
    }
    // try to set the desired state
    try {
        mTeleopState.SetDesiredState(state);
    } catch (...) {
        mInterface->SendError(this->GetName() + ": " + state + " is not an allowed desired state");
        return;
    }
    // force operator to indicate they are present
    m_operator.is_active = false;
    MessageEvents.desired_state(state);
    mInterface->SendStatus(this->GetName() + ": set desired state to " + state);
}

vctMatRot3 mtsTeleOperationPSM::UpdateAlignOffset(void)
{
    vctMatRot3 desiredOrientation = mPSM.m_setpoint_cp.Position().Rotation();
    mMTM.m_measured_cp.Position().Rotation().ApplyInverseTo(desiredOrientation, m_alignment_offset);
    return desiredOrientation;
}

void mtsTeleOperationPSM::UpdateInitialState(void)
{
    mMTM.CartesianInitial.From(mMTM.m_measured_cp.Position());
    mMTM.m_initial_cp = mMTM.m_measured_cp;
    mPSM.CartesianInitial.From(mPSM.m_setpoint_cp.Position());
    mPSM.m_initial_cp = mPSM.m_setpoint_cp;

    UpdateAlignOffset();
    m_alignment_offset_initial = m_alignment_offset;
    if (mBaseFrame.measured_cp.IsValid()) {
        mBaseFrame.CartesianInitial.From(mBaseFrame.m_measured_cp.Position());
    }
}

void mtsTeleOperationPSM::set_scale(const double & scale)
{
    // set scale
    mConfigurationStateTable->Start();
    m_scale = scale;
    mConfigurationStateTable->Advance();
    ConfigurationEvents.scale(m_scale);

    // update MTM/PSM previous position to prevent jumps
    UpdateInitialState();
}

void mtsTeleOperationPSM::lock_rotation(const bool & lock)
{
    mConfigurationStateTable->Start();
    m_rotation_locked = lock;
    mConfigurationStateTable->Advance();
    ConfigurationEvents.rotation_locked(m_rotation_locked);
    // when releasing the orientation, MTM orientation is likely off
    // so force re-align
    if (lock == false) {
        set_following(false);
        mTeleopState.SetCurrentState("DISABLED");
    } else {
        // update MTM/PSM previous position
        UpdateInitialState();
        // lock orientation if the arm is running
        if ((mTeleopState.CurrentState() == "ENABLED")
            && mMTM.lock_orientation.IsValid()) {
            mMTM.lock_orientation(mMTM.m_measured_cp.Position().Rotation());
        }
    }
}

void mtsTeleOperationPSM::lock_translation(const bool & lock)
{
    mConfigurationStateTable->Start();
    m_translation_locked = lock;
    mConfigurationStateTable->Advance();
    ConfigurationEvents.translation_locked(m_translation_locked);
    // update MTM/PSM previous position
    UpdateInitialState();
}

void mtsTeleOperationPSM::set_align_mtm(const bool & alignMTM)
{
    mConfigurationStateTable->Start();
    // make sure we have access to lock/unlock
    if ((mMTM.lock_orientation.IsValid()
         && mMTM.unlock_orientation.IsValid())) {
        m_align_mtm = alignMTM;
    } else {
        if (alignMTM) {
            mInterface->SendWarning(this->GetName() + ": unable to force MTM alignment, the device doesn't provide commands to lock/unlock orientation");
        }
        m_align_mtm = false;
    }
    mConfigurationStateTable->Advance();
    ConfigurationEvents.align_mtm(m_align_mtm);
    // force re-align if the teleop is already enabled
    if (mTeleopState.CurrentState() == "ENABLED") {
        mTeleopState.SetCurrentState("DISABLED");
    }
}

void mtsTeleOperationPSM::StateChanged(void)
{
    const std::string newState = mTeleopState.CurrentState();
    MessageEvents.current_state(newState);
    mInterface->SendStatus(this->GetName() + ": current state is " + newState);
}

void mtsTeleOperationPSM::RunAllStates(void)
{
    Result result;

    result = mMTM.getData();
    if (!result.ok) {
        CMN_LOG_CLASS_RUN_ERROR << "Run: MTM: " << result.message << std::endl;
        mInterface->SendError(this->GetName() + ": MTM: " + result.message);
        mTeleopState.SetDesiredState("DISABLED");
    }

    result = mPSM.getData();
    if (!result.ok) {
        CMN_LOG_CLASS_RUN_ERROR << "Run: PSM: " << result.message << std::endl;
        mInterface->SendError(this->GetName() + ": PSM: " + result.message);
        mTeleopState.SetDesiredState("DISABLED");
    }

    mtsExecutionResult executionResult;
    // get base-frame cartesian position if available
    if (mBaseFrame.measured_cp.IsValid()) {
        executionResult = mBaseFrame.measured_cp(mBaseFrame.m_measured_cp);
        if (!executionResult.IsOK()) {
            CMN_LOG_CLASS_RUN_ERROR << "Run: call to m_base_frame.measured_cp failed \""
                                    << executionResult << "\"" << std::endl;
            mInterface->SendError(this->GetName() + ": unable to get cartesian position from base frame");
            mTeleopState.SetDesiredState("DISABLED");
        }
    }

    // check if anyone wanted to disable anyway
    if ((mTeleopState.DesiredState() == "DISABLED")
        && (mTeleopState.CurrentState() != "DISABLED")) {
        set_following(false);
        mTeleopState.SetCurrentState("DISABLED");
        return;
    }

    // monitor state of arms if needed
    if ((mTeleopState.CurrentState() != "DISABLED")
        && (mTeleopState.CurrentState() != "SETTING_ARMS_STATE")) {
        prmOperatingState state;
        mPSM.operating_state(state);
        if ((state.State() != prmOperatingState::ENABLED)
            || !state.IsHomed()) {
            mTeleopState.SetDesiredState("DISABLED");
            mInterface->SendError(this->GetName() + ": PSM is not in state \"ENABLED\" anymore");
        }
        mMTM.operating_state(state);
        if ((state.State() != prmOperatingState::ENABLED)
            || !state.IsHomed()) {
            mTeleopState.SetDesiredState("DISABLED");
            mInterface->SendError(this->GetName() + ": MTM is not in state \"READY\" anymore");
        }
    }
}

void mtsTeleOperationPSM::TransitionDisabled(void)
{
    if (mTeleopState.DesiredStateIsNotCurrent()) {
        mTeleopState.SetCurrentState("SETTING_ARMS_STATE");
    }
}

void mtsTeleOperationPSM::EnterSettingArmsState(void)
{
    // reset timer
    mInStateTimer = StateTable.GetTic();

    // request state if needed
    prmOperatingState state;
    mPSM.operating_state(state);
    if (state.State() != prmOperatingState::ENABLED) {
        mPSM.state_command(std::string("enable"));
    }
    if (!state.IsHomed()) {
        mPSM.state_command(std::string("home"));
    }

    mMTM.operating_state(state);
    if (state.State() != prmOperatingState::ENABLED) {
        mMTM.state_command(std::string("enable"));
    }
    if (!state.IsHomed()) {
        mMTM.state_command(std::string("home"));
    }
}

void mtsTeleOperationPSM::TransitionSettingArmsState(void)
{
    // check state
    prmOperatingState psmState, mtmState;
    mPSM.operating_state(psmState);
    mMTM.operating_state(mtmState);
    if ((psmState.State() == prmOperatingState::ENABLED) && psmState.IsHomed()
        && (mtmState.State() == prmOperatingState::ENABLED) && mtmState.IsHomed()) {
        mTeleopState.SetCurrentState("ALIGNING_MTM");
        return;
    }
    // check timer
    if ((StateTable.GetTic() - mInStateTimer) > 60.0 * cmn_s) {
        if (!((psmState.State() == prmOperatingState::ENABLED) && psmState.IsHomed())) {
            mInterface->SendError(this->GetName() + ": timed out while setting up PSM state");
        }
        if (!((mtmState.State() == prmOperatingState::ENABLED) && mtmState.IsHomed())) {
            mInterface->SendError(this->GetName() + ": timed out while setting up MTM state");
        }
        mTeleopState.SetDesiredState("DISABLED");
    }
}

void mtsTeleOperationPSM::EnterAligningMTM(void)
{
    // update user GUI re. scale
    ConfigurationEvents.scale(m_scale);

    // reset timer
    mInStateTimer = StateTable.GetTic();
    mTimeSinceLastAlign = 0.0;

    // if we don't align MTM, just stay in same position
    if (!m_align_mtm) {
        // convert to prm type
        mMTM.m_move_cp.Goal().Assign(mMTM.m_setpoint_cp.Position());
        mMTM.move_cp(mMTM.m_move_cp);
    }

    if (m_back_from_clutch) {
        m_operator.is_active = m_operator.was_active_before_clutch;
        m_back_from_clutch = false;
    }

    if (!m_jaw.ignore) {
        // figure out the mapping between the MTM gripper angle and the PSM jaw angle
        UpdateGripperToJawConfiguration();
    }

    // set min/max for roll outside bounds
    m_operator.roll_min = cmnPI * 100.0;
    m_operator.roll_max = -cmnPI * 100.0;
    m_operator.gripper_min = cmnPI * 100.0;
    m_operator.gripper_max = -cmnPI * 100.0;
}

void mtsTeleOperationPSM::RunAligningMTM(void)
{
    // if clutched or align not needed, do nothing
    if (m_clutched || !m_align_mtm) {
        return;
    }

    // set trajectory goal periodically, this will track PSM motion
    const double currentTime = StateTable.GetTic();
    if ((currentTime - mTimeSinceLastAlign) > 10.0 * cmn_ms) {
        mTimeSinceLastAlign = currentTime;
        // Orientate MTM with PSM
        vctFrm4x4 mtmCartesianGoal;
        mtmCartesianGoal.Translation().Assign(mMTM.m_setpoint_cp.Position().Translation());
        mtmCartesianGoal.Rotation().FromNormalized(mPSM.m_setpoint_cp.Position().Rotation());
        // convert to prm type
        mMTM.m_move_cp.Goal().From(mtmCartesianGoal);
        mMTM.move_cp(mMTM.m_move_cp);
    }
}

void mtsTeleOperationPSM::TransitionAligningMTM(void)
{
    // if the desired state is aligning MTM, just stay here
    if (!mTeleopState.DesiredStateIsNotCurrent()) {
        return;
    }

    // check difference of orientation between mtm and PSM to enable
    vctMatRot3 desiredOrientation = UpdateAlignOffset();
    vctAxAnRot3 axisAngle(m_alignment_offset, VCT_NORMALIZE);
    double orientationError = 0.0;
    // set error only if we need to align MTM to PSM
    if (m_align_mtm) {
        orientationError = axisAngle.Angle();
    }

    // if not active, use gripper and/or roll to detect if the user is ready
    if (!m_operator.is_active) {
        // update gripper values
        double gripperRange = 0.0;
        if (mMTM.gripper_measured_js.IsValid()) {
            mMTM.gripper_measured_js(mMTM.m_gripper_measured_js);
            const double gripper = mMTM.m_gripper_measured_js.Position()[0];
            if (gripper > m_operator.gripper_max) {
                m_operator.gripper_max = gripper;
            } else if (gripper < m_operator.gripper_min) {
                m_operator.gripper_min = gripper;
            }
            gripperRange = m_operator.gripper_max - m_operator.gripper_min;
        }

        // checking roll
        const double roll = acos(vctDotProduct(desiredOrientation.Column(1),
                                               mMTM.m_measured_cp.Position().Rotation().Column(1)));
        if (roll > m_operator.roll_max) {
            m_operator.roll_max = roll;
        } else if (roll < m_operator.roll_min) {
            m_operator.roll_min = roll;
        }
        const double rollRange = m_operator.roll_max - m_operator.roll_min;

        // different conditions to set operator active
        if (gripperRange >= m_operator.gripper_threshold) {
            m_operator.is_active = true;
        } else if (rollRange >= m_operator.roll_threshold) {
            m_operator.is_active = true;
        } else if ((gripperRange + rollRange)
                   > 0.8 * (m_operator.gripper_threshold
                            + m_operator.roll_threshold)) {
            m_operator.is_active = true;
        }
    }

    // finally check for transition
    if ((orientationError <= m_operator.orientation_tolerance)
        && m_operator.is_active) {
        if (mTeleopState.DesiredState() == "ENABLED") {
            mTeleopState.SetCurrentState("ENABLED");
        }
    } else {
        // check timer and issue a message
        if ((StateTable.GetTic() - mInStateTimer) > 2.0 * cmn_s) {
            std::stringstream message;
            if (orientationError >= m_operator.orientation_tolerance) {
                message << this->GetName() + ": unable to align MTM, angle error is "
                        << orientationError * cmn180_PI << " (deg)";
            } else if (!m_operator.is_active) {
                message << this->GetName() + ": pinch/twist MTM gripper a bit";
            }
            mInterface->SendWarning(message.str());
            mInStateTimer = StateTable.GetTic();
        }
    }
}

void mtsTeleOperationPSM::EnterEnabled(void)
{
    // update MTM/PSM previous position
    UpdateInitialState();

    // set gripper ghost if needed
    if (!m_jaw.ignore) {
        m_jaw_caught_up_after_clutch = false;
        // gripper ghost
        mPSM.jaw_measured_js(mPSM.m_jaw_setpoint_js);
        if (mPSM.m_jaw_setpoint_js.Position().size() != 1) {
            mInterface->SendWarning(this->GetName() + ": unable to get jaw position.  Make sure there is an instrument on the PSM");
            mTeleopState.SetDesiredState("DISABLE");
        }
        double currentJaw = mPSM.m_jaw_setpoint_js.Position()[0];
        m_gripper_ghost = JawToGripper(currentJaw);
    }

    // orientation locked or not
    if (m_rotation_locked
        && mMTM.lock_orientation.IsValid()) {
        mMTM.lock_orientation(mMTM.m_measured_cp.Position().Rotation());
    } else {
        if (mMTM.unlock_orientation.IsValid()) {
            mMTM.unlock_orientation();
        }
    }

    // check if by any chance the clutch pedal is pressed
    if (m_clutched) {
        Clutch(true);
    } else {
        set_following(true);
    }

    psm_js_data = {};
    mtm_js_data = {};
}

void mtsTeleOperationPSM::UnilateralTeleop() {
    mPSM.m_servo_cpvf = mPSM.computeGoalFromTarget(&mMTM, m_alignment_offset_initial, m_scale);
    mPSM.m_servo_cpvf.EffortIsDefined() = false;
    mPSM.servo_cpvf(mPSM.m_servo_cpvf);
}

void mtsTeleOperationPSM::BilateralTeleop() {
    mPSM.m_servo_cpvf = mPSM.computeGoalFromTarget(&mMTM, m_alignment_offset_initial, m_scale);
    mMTM.m_servo_cpvf = mMTM.computeGoalFromTarget(&mPSM, m_alignment_offset_initial.Inverse(), 1.0 / m_scale);
    
    mMTM.servo_cpvf(mMTM.m_servo_cpvf);
    mPSM.servo_cpvf(mPSM.m_servo_cpvf);
}

void mtsTeleOperationPSM::HighLatencyTeleop() {
    mPSM.m_servo_cpvf = mPSM.computeGoalFromTarget(&mMTM, m_alignment_offset_initial, m_scale);
    mMTM.m_servo_cpvf = mMTM.computeGoalFromTarget(&mPSM, m_alignment_offset_initial.Inverse(), 1.0 / m_scale);

    mPSM.m_servo_cpvf.PositionIsDefined() = false;
    mPSM.m_servo_cpvf.VelocityIsDefined() = false;
    mMTM.m_servo_cpvf.EffortIsDefined() = false;

    mMTM.servo_cpvf(mMTM.m_servo_cpvf);
    mPSM.servo_cpvf(mPSM.m_servo_cpvf);
}

void mtsTeleOperationPSM::RunEnabled(void)
{
    if (m_clutched) {
        return;
    }

    if (!mMTM.m_measured_cp.Valid() || !mMTM.m_setpoint_cp.Valid()) {
        return;
    }

    if (!mPSM.m_measured_cp.Valid() || !mPSM.m_setpoint_cp.Valid()) {
        return;
    }

    switch (mTeleopMode)
    {
    case Mode::HIGH_LATENCY:
        HighLatencyTeleop();
        break;
    case Mode::BILATERAL:
        BilateralTeleop();
        break;
    case Mode::UNILATERAL:
    default:
        UnilateralTeleop();
        break;
    }

    vct6 psm_jp;
    vct6 psm_jv;
    vct6 psm_jf;

    psm_jp.Assign(mPSM.m_measured_js.Position().Ref(6));
    psm_jv.Assign(mPSM.m_measured_js.Velocity().Ref(6));
    psm_jf.Assign(mPSM.m_measured_js.Effort().Ref(6));

    psm_js_data.emplace_back(psm_jp, psm_jv, psm_jf);

    vct7 mtm_jp;
    vct7 mtm_jv;
    vct7 mtm_jf;

    mtm_jp.Assign(mMTM.m_measured_js.Position().Ref(7));
    mtm_jv.Assign(mMTM.m_measured_js.Velocity().Ref(7));
    mtm_jf.Assign(mMTM.m_measured_js.Effort().Ref(7));

    mtm_js_data.emplace_back(mtm_jp, mtm_jv, mtm_jf);

    if (m_jaw.ignore) {
        return;
    }

    // open jaws to 45 degrees if we don't have MTM gripper position
    if (!mMTM.gripper_measured_js.IsValid()) {
        mPSM.m_jaw_servo_jp.Goal()[0] = 45.0 * cmnPI_180;
        mPSM.jaw_servo_jp(mPSM.m_jaw_servo_jp);
    } else {
        const double currentGripper = mMTM.m_gripper_measured_js.Position()[0];
        // see if we caught up
        if (!m_jaw_caught_up_after_clutch) {
            const double error = std::abs(currentGripper - m_gripper_ghost);
            if (error < mtsIntuitiveResearchKit::TeleOperationPSM::ToleranceBackFromClutch) {
                m_jaw_caught_up_after_clutch = true;
            }
        }

        // pick the rate based on back from clutch or not
        const double delta = m_jaw_caught_up_after_clutch ?
            m_jaw.rate * StateTable.PeriodStats.PeriodAvg()
            : m_jaw.rate_back_from_clutch * StateTable.PeriodStats.PeriodAvg();
        const double error = std::min(delta, std::fabs(currentGripper - m_gripper_ghost));
        m_gripper_ghost += std::copysign(error, currentGripper - m_gripper_ghost);
        m_gripper_ghost = std::max(m_gripper_ghost, m_gripper_to_jaw.position_min);
        mPSM.m_jaw_servo_jp.Goal()[0] = GripperToJaw(m_gripper_ghost);
        mPSM.jaw_servo_jp(mPSM.m_jaw_servo_jp);
    }
}

void write_header(std::ostream& file, int joints) {
    for (int j = 0; j < joints; j++) {
            file << "jp_" << j << ",";
    }

    for (int j = 0; j < joints; j++) {
        file << "jv_" << j << ",";
    }

    for (int j = 0; j < joints; j++) {
        file << "jf_" << j;
        if (j == joints-1) {
            file << "\n";
        } else {
            file << ",";
        }
    }
}

template <int joints>
void save_js_data(std::string output_file, const std::vector<std::tuple<vctFixedSizeVector<double, joints>, vctFixedSizeVector<double, joints>, vctFixedSizeVector<double, joints>>>& data) {
    std::ofstream file(output_file);
    write_header(file, joints);

    for (auto& row : data) {
        for (int j = 0; j < joints; j++) {
            file << (std::get<0>(row))[j] << ",";
        }
        
        for (int j = 0; j < joints; j++) {
            file << (std::get<1>(row))[j] << ",";
        }

        for (int j = 0; j < joints; j++) {
            file << (std::get<2>(row))[j];
            if (j < 5) {
                file << ",";
            }
        }

        file << "\n";
    }
}

void mtsTeleOperationPSM::TransitionEnabled(void)
{
    if (mTeleopState.DesiredStateIsNotCurrent()) {
        set_following(false);
        mTeleopState.SetCurrentState(mTeleopState.DesiredState());

        mPSM.m_servo_cpvf.PositionIsDefined() = false;
        mPSM.m_servo_cpvf.VelocityIsDefined() = false;
        mPSM.m_servo_cpvf.EffortIsDefined() = false;

        mMTM.m_servo_cpvf.PositionIsDefined() = false;
        mMTM.m_servo_cpvf.VelocityIsDefined() = false;
        mMTM.m_servo_cpvf.EffortIsDefined() = false;

        mPSM.servo_cpvf(mPSM.m_servo_cpvf);
        mMTM.servo_cpvf(mMTM.m_servo_cpvf);

        save_js_data<6>(js_data_output_folder + "psm_js.csv", psm_js_data);
        save_js_data<7>(js_data_output_folder + "mtm_js.csv", mtm_js_data);
    }
}

double mtsTeleOperationPSM::GripperToJaw(const double & gripperAngle) const
{
    return m_gripper_to_jaw.scale * gripperAngle + m_gripper_to_jaw.offset;
}

double mtsTeleOperationPSM::JawToGripper(const double & jawAngle) const
{
    return (jawAngle - m_gripper_to_jaw.offset) / m_gripper_to_jaw.scale;
}

void mtsTeleOperationPSM::UpdateGripperToJawConfiguration(void)
{
    // default values, assumes jaws match gripper
    double _jaw_min = 0.0;
    double _jaw_max = m_gripper.max;

    m_gripper_to_jaw.position_min = 0.0;
    // get the PSM jaw configuration if possible to find range
    if (mPSM.jaw_configuration_js.IsValid()) {
        mPSM.jaw_configuration_js(mPSM.m_jaw_configuration_js);
        if ((mPSM.m_jaw_configuration_js.PositionMin().size() == 1)
            && (mPSM.m_jaw_configuration_js.PositionMax().size() == 1)) {
            _jaw_min = mPSM.m_jaw_configuration_js.PositionMin()[0];
            _jaw_max = mPSM.m_jaw_configuration_js.PositionMax()[0];
            // save min for later so we never ask PSM to close jaws more than min
            m_gripper_to_jaw.position_min = _jaw_min;
        }
    }
    // if the PSM can close its jaws past 0 (tighter), we map from 0 to qmax
    // negative values just mean tighter jaws
    m_gripper_to_jaw.scale = (_jaw_max) / (m_gripper.max - m_gripper.zero);
    m_gripper_to_jaw.offset = -m_gripper.zero / m_gripper_to_jaw.scale;
}

void mtsTeleOperationPSM::set_following(const bool following)
{
    MessageEvents.following(following);
    m_following = following;
}
