/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  Author(s):  Zihan Chen, Anton Deguet
  Created on: 2013-03-06

  (C) Copyright 2013-2023 Johns Hopkins University (JHU), All Rights Reserved.

  --- begin cisst license - do not edit ---

  This software is provided "as is" under an open source license, with
  no warranty.  The complete license can be found in license.txt and
  http://www.cisst.org/cisst/license.txt.

  --- end cisst license ---
*/

#ifndef _mtsTeleOperationPSM_h
#define _mtsTeleOperationPSM_h

#include <vector>
#include <tuple>
#include <string>
#include <deque>

#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstParameterTypes/prmEventButton.h>
#include <cisstParameterTypes/prmPositionCartesianGet.h>
#include <cisstParameterTypes/prmVelocityCartesianGet.h>
#include <cisstParameterTypes/prmForceCartesianGet.h>
#include <cisstParameterTypes/prmPositionCartesianSet.h>
#include <cisstParameterTypes/prmForceCartesianSet.h>
#include <cisstParameterTypes/prmStateJoint.h>
#include <cisstParameterTypes/prmConfigurationJoint.h>
#include <cisstParameterTypes/prmPositionJointSet.h>
#include <cisstParameterTypes/prmStateCartesian.h>

#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKit.h>
#include <sawIntuitiveResearchKit/mtsStateMachine.h>

// always include last
#include <sawIntuitiveResearchKit/sawIntuitiveResearchKitExport.h>

class CISST_EXPORT mtsTeleOperationPSM: public mtsTaskPeriodic
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION_ONEARG, CMN_LOG_ALLOW_DEFAULT);

 public:
    enum class Mode {
        UNILATERAL,
        BILATERAL,
        HIGH_LATENCY
    };

    mtsTeleOperationPSM(const std::string & componentName, const double periodInSeconds);
    mtsTeleOperationPSM(const mtsTaskPeriodicConstructorArg & arg);
    ~mtsTeleOperationPSM();

    void Configure(const std::string & filename = "");
    virtual void Configure(const Json::Value & jsonConfig);
    void Startup(void);
    void Run(void);
    void Cleanup(void);

    void set_scale(const double & scale);
    void lock_rotation(const bool & lock);
    void lock_translation(const bool & lock);
    void set_align_mtm(const bool & alignMTM);
    void following_mtm_body_servo_cf(const prmForceCartesianSet & wrench);

 protected:

    virtual void Init(void);

    // Event Handler
    void MTMErrorEventHandler(const mtsMessage & message);
    void PSMErrorEventHandler(const mtsMessage & message);

    void ClutchEventHandler(const prmEventButton & button);
    void Clutch(const bool & clutch);

    // Functions for events
    struct {
        mtsFunctionWrite desired_state;
        mtsFunctionWrite current_state;
        mtsFunctionWrite following;
    } MessageEvents;
    mtsInterfaceProvided * mInterface;

    struct {
        mtsFunctionWrite scale;
        mtsFunctionWrite rotation_locked;
        mtsFunctionWrite translation_locked;
        mtsFunctionWrite align_mtm;
    } ConfigurationEvents;

    void SetDesiredState(const std::string & state);
    void state_command(const std::string & command);

    vctMatRot3 UpdateAlignOffset(void);
    void UpdateInitialState(void);

    void StateChanged(void);
    void RunAllStates(void); // this should happen for all states
    void TransitionDisabled(void); // checks for desired state
    void EnterSettingArmsState(void);
    void TransitionSettingArmsState(void);
    void EnterAligningMTM(void);
    void RunAligningMTM(void);
    void TransitionAligningMTM(void);
    void EnterEnabled(void); // called when enabling, save initial positions of master and slave
    void RunEnabled(void); // performs actual teleoperation
    void TransitionEnabled(void); // performs actual teleoperation

    class Result {
    public:
        bool ok;
        std::string message;

        Result() : ok(true), message("") {}
        Result(bool ok, std::string message, mtsExecutionResult executionResult);
    };

    // Light wrapper around common arm functionality
    class Arm {
    public:
        virtual ~Arm() {};

        virtual void populateInterface(mtsInterfaceRequired* interfaceRequired);
        virtual prmStateCartesian computeGoalFromTarget(Arm* target, const vctMatRot3& alignment_offset, double size_scale) const;

        virtual Result getData();

        mtsFunctionRead  operating_state;
        mtsFunctionWrite state_command;

        mtsFunctionRead  measured_cp;
        mtsFunctionRead  measured_js;
        mtsFunctionRead  body_measured_cv;
        mtsFunctionRead  body_measured_cf;
        mtsFunctionRead  setpoint_cp;

        mtsFunctionWrite servo_cpvf;

        mtsFunctionWrite use_gravity_compensation;

        vctFrm4x4 CartesianInitial;

        prmPositionCartesianGet m_initial_cp;

        prmPositionCartesianGet m_measured_cp;
        prmVelocityCartesianGet m_body_measured_cv;
        prmForceCartesianGet    m_body_measured_cf;
        prmStateJoint           m_measured_js;
        prmPositionCartesianGet m_setpoint_cp;
        prmStateCartesian       m_servo_cpvf;

        double sampleDelay = 0.0;
        std::deque<prmPositionCartesianGet> cp_delay_buffer;
        std::deque<prmVelocityCartesianGet> cv_delay_buffer;
        std::deque<prmForceCartesianGet> cf_delay_buffer;
    };

    class ArmMTM : public Arm {
    public:
        void populateInterface(mtsInterfaceRequired* interfaceRequired) override;

        Result getData() override;

        mtsFunctionWrite move_cp;
        mtsFunctionRead  gripper_measured_js;

        mtsFunctionWrite lock_orientation;
        mtsFunctionVoid  unlock_orientation;
        mtsFunctionWrite body_servo_cf;

        prmPositionCartesianSet m_move_cp;
        prmStateJoint m_gripper_measured_js;

        std::deque<prmStateJoint> gripper_delay_buffer;
    };

    class ArmPSM : public Arm {
    public:
        void populateInterface(mtsInterfaceRequired* interfaceRequired) override;

        Result getData() override;

        mtsFunctionVoid  hold;
        mtsFunctionRead  jaw_setpoint_js;
        mtsFunctionRead  jaw_configuration_js;
        mtsFunctionWrite jaw_servo_jp;

        prmStateJoint m_jaw_setpoint_js;
        prmConfigurationJoint m_jaw_configuration_js;
        prmPositionJointSet m_jaw_servo_jp;
    };

    Mode mTeleopMode;
    ArmMTM mMTM;
    ArmPSM mPSM;

    void UnilateralTeleop();
    void BilateralTeleop();
    void HighLatencyTeleop();

    struct {
        mtsFunctionRead  measured_cp;
        prmPositionCartesianGet m_measured_cp;
        vctFrm4x4 CartesianInitial;
    } mBaseFrame;

    double m_scale = mtsIntuitiveResearchKit::TeleOperationPSM::Scale;
    vctMatRot3 m_alignment_offset,
        m_alignment_offset_initial; // rotation offset between MTM and PSM when tele-operation goes in follow mode

    // conversion from gripper (MTM) to jaw (PSM)
    // j = s * g + o
    // g = (j - o) / s
    struct {
        double scale;
        double offset;
        double position_min;
    } m_gripper_to_jaw;

    double m_gripper_ghost;

    double virtual GripperToJaw(const double & gripperAngle) const;
    double virtual JawToGripper(const double & jawAngle) const;
    void virtual UpdateGripperToJawConfiguration(void);

    struct {
        bool ignore = false; // flag to tele-op in cartesian position only, don't need or drive the PSM jaws
        double rate = mtsIntuitiveResearchKit::TeleOperationPSM::JawRate;
        double rate_back_from_clutch = mtsIntuitiveResearchKit::TeleOperationPSM::JawRateBackFromClutch;
    } m_jaw;

    struct {
        double max = 60.0 * cmnPI_180; // from 2012, we assumed MTM gripper is 0 to 60 degrees
        double zero = 0.0; // value corresponding to closed SPM jaws,
                           // MTM gripper might go lower to force negative
                           // PSM jaw PID goal and stronger forces
    } m_gripper;

    struct {
        double orientation_tolerance = mtsIntuitiveResearchKit::TeleOperationPSM::OrientationTolerance;
        double roll_min;
        double roll_max;
        double roll_threshold = mtsIntuitiveResearchKit::TeleOperationPSM::RollThreshold;
        double gripper_min;
        double gripper_max;
        double gripper_threshold = mtsIntuitiveResearchKit::TeleOperationPSM::GripperThreshold;
        bool is_active = false;
        bool was_active_before_clutch = false;
    } m_operator;

    bool m_clutched = false;
    bool m_back_from_clutch = false;
    bool m_jaw_caught_up_after_clutch = false;
    bool m_rotation_locked = false;
    bool m_translation_locked = false;
    bool m_align_mtm = true; // default on da Vinci
    prmForceCartesianSet m_following_mtm_body_servo_cf;

    vctMatRot3 mMTMClutchedOrientation;
    mtsStateTable * mConfigurationStateTable;

    mtsStateMachine mTeleopState;
    double mInStateTimer;
    double mTimeSinceLastAlign;

    bool m_following;
    void set_following(const bool following);

    std::vector<std::tuple<vct6, vct6, vct6>> force_data;
    std::string force_data_output_file = "/home/bburkha4/catkin_ws/src/dvrk-ros/dvrk_python/scripts/data/psm_js_test.csv";
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTeleOperationPSM);

#endif // _mtsTeleOperationPSM_h
