#ifndef mtsRemoteMTM_h
#define mtsRemoteMTM_h

#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>
#include <sawIntuitiveResearchKit/mtsIntuitiveResearchKit.h>

#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/WrenchStamped.h>

#include <crtk_msgs/CartesianState.h>
#include <crtk_msgs/OperatingState.h>
#include <crtk_msgs/StringStamped.h>

#include <cisstParameterTypes/prmOperatingState.h>
#include <cisstParameterTypes/prmStateCartesian.h>
#include <cisstParameterTypes/prmStateJoint.h>
#include <cisstParameterTypes/prmPositionCartesianGet.h>
#include <cisstParameterTypes/prmPositionCartesianSet.h>
#include <cisstParameterTypes/prmVelocityCartesianGet.h>
#include <cisstParameterTypes/prmForceCartesianGet.h>

class mtsRemoteMTM : public mtsTaskPeriodic
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION_ONEARG, CMN_LOG_ALLOW_DEFAULT);

    friend class mtsIntuitiveResearchKitConsole;

public:
    mtsRemoteMTM(const std::string & componentName, const double periodInSeconds = mtsIntuitiveResearchKit::ArmPeriod);
    mtsRemoteMTM(const mtsTaskPeriodicConstructorArg & arg);

    virtual ~mtsRemoteMTM() {};
    
    void Configure(const std::string & CMN_UNUSED(filename) = "") override {};
    void Run(void) override;
    void Startup(void) {};
    void Cleanup(void) {};

    void Init(void);

private:
    ros::NodeHandle node_handle;

    ros::Publisher state_command_pub;
    ros::Publisher servo_cpvf_pub;
    ros::Publisher move_cp_pub;
    ros::Publisher lock_orientation_pub;
    ros::Publisher unlock_orientation_pub;

    ros::Subscriber operating_state_sub;
    ros::Subscriber measured_js_sub;
    ros::Subscriber setpoint_js_sub;
    ros::Subscriber gripper_measured_js_sub;
    ros::Subscriber measured_cp_sub;
    ros::Subscriber setpoint_cp_sub;
    ros::Subscriber body_measured_cv_sub;
    ros::Subscriber body_measured_cf_sub;
    ros::Subscriber body_external_cf_sub;

    mtsInterfaceProvided* m_arm_interface;

    prmOperatingState operating_state;

    prmStateJoint measured_js;
    prmStateJoint setpoint_js;
    prmStateJoint gripper_measured_js;

    prmPositionCartesianGet measured_cp;
    prmPositionCartesianGet setpoint_cp;

    prmVelocityCartesianGet measured_cv;
    prmForceCartesianGet body_measured_cf;
    prmForceCartesianGet body_external_cf;

    void InitializeStateTable();
    void InitializePubSub(std::string ros_prefix);
    void InitializeArmInterface();

    void StateCommand(const std::string & command);
    void ServoCPVF(const prmStateCartesian & cs);
    void MoveCP(const prmPositionCartesianSet & move_cp);
    void LockOrientation(const vctMatRot3 & orientation);
    void UnlockOrientation(void);

    void OperatingStateCallback(const crtk_msgs::OperatingState& msg);
    void MeasuredJSCallback(const sensor_msgs::JointState& msg);
    void SetpointJSCallback(const sensor_msgs::JointState& msg);
    void GripperMeasuredJSCallback(const sensor_msgs::JointState& msg);
    void MeasuredCPCallback(const geometry_msgs::PoseStamped& msg);
    void SetpointCPCallback(const geometry_msgs::PoseStamped& msg);
    void MeasuredCVCallback(const geometry_msgs::TwistStamped& msg);
    void MeasuredCFCallback(const geometry_msgs::WrenchStamped& msg);
    void ExternalCFCallback(const geometry_msgs::WrenchStamped& msg);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsRemoteMTM);

#endif
