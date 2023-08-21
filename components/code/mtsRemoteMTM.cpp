#include <sawIntuitiveResearchKit/mtsRemoteMTM.h>

#include <cisstMultiTask/mtsInterfaceProvided.h>

CMN_IMPLEMENT_SERVICES_DERIVED_ONEARG(mtsRemoteMTM, mtsTaskPeriodic, mtsTaskPeriodicConstructorArg);

mtsRemoteMTM::mtsRemoteMTM(const std::string & componentName, const double periodInSeconds)
 : mtsTaskPeriodic(componentName, periodInSeconds)
{
    Init();
}

mtsRemoteMTM::mtsRemoteMTM(const mtsTaskPeriodicConstructorArg & arg):
    mtsTaskPeriodic(arg)
{
    Init();
}

void mtsRemoteMTM::InitializeStateTable() {
    operating_state.SetAutomaticTimestamp(false);
    this->StateTable.AddData(operating_state, "operating_state");

    measured_js.SetAutomaticTimestamp(false);
    this->StateTable.AddData(measured_js, "measured_js");

    setpoint_js.SetAutomaticTimestamp(false);
    this->StateTable.AddData(setpoint_js, "setpoint_js");

    measured_cp.SetAutomaticTimestamp(false);
    this->StateTable.AddData(measured_cp, "measured_cp");

    setpoint_cp.SetAutomaticTimestamp(false);
    this->StateTable.AddData(setpoint_cp, "setpoint_cp");

    measured_cv.SetAutomaticTimestamp(false);
    this->StateTable.AddData(measured_cv, "measured_cv");

    body_measured_cf.SetAutomaticTimestamp(false); 
    this->StateTable.AddData(body_measured_cf, "body/measured_cf");

    body_external_cf.SetAutomaticTimestamp(false); 
    this->StateTable.AddData(body_external_cf, "external/measured_cf");

    gripper_measured_js.SetAutomaticTimestamp(false); 
    this->StateTable.AddData(gripper_measured_js, "gripper/measured_js");
}

void mtsRemoteMTM::InitializePubSub(std::string ros_prefix)
{
    state_command_pub = node_handle.advertise<crtk_msgs::StringStamped>(ros_prefix + "state_command", 10);
    servo_cpvf_pub = node_handle.advertise<crtk_msgs::CartesianState>(ros_prefix + "servo_cpvf", 10);
    move_cp_pub = node_handle.advertise<geometry_msgs::PoseStamped>(ros_prefix + "move_cp", 10);
    lock_orientation_pub = node_handle.advertise<geometry_msgs::Quaternion>(ros_prefix + "lock_orientation", 10);
    unlock_orientation_pub = node_handle.advertise<std_msgs::Empty>(ros_prefix + "unlock_orientation", 10);

    operating_state_sub = node_handle.subscribe(ros_prefix + "operating_state", 10, &mtsRemoteMTM::OperatingStateCallback, this);
    measured_js_sub = node_handle.subscribe(ros_prefix + "measured_js", 10, &mtsRemoteMTM::MeasuredJSCallback, this);
    setpoint_js_sub = node_handle.subscribe(ros_prefix + "setpoint_js", 10, &mtsRemoteMTM::SetpointJSCallback, this);
    gripper_measured_js_sub = node_handle.subscribe(ros_prefix + "gripper/measured_js", 10, &mtsRemoteMTM::GripperMeasuredJSCallback, this);
    measured_cp_sub = node_handle.subscribe(ros_prefix + "measured_cp", 10, &mtsRemoteMTM::MeasuredCPCallback, this);
    setpoint_cp_sub = node_handle.subscribe(ros_prefix + "setpoint_cp", 10, &mtsRemoteMTM::SetpointCPCallback, this);
    body_measured_cv_sub = node_handle.subscribe(ros_prefix + "measured_cv", 10, &mtsRemoteMTM::MeasuredCVCallback, this);
    body_measured_cf_sub = node_handle.subscribe(ros_prefix + "body/measured_cf", 10, &mtsRemoteMTM::MeasuredCFCallback, this);
    body_external_cf_sub = node_handle.subscribe(ros_prefix + "external/measured_cf", 10, &mtsRemoteMTM::ExternalCFCallback, this);
}

void mtsRemoteMTM::InitializeArmInterface()
{
    m_arm_interface = AddInterfaceProvided("Arm");
    if (m_arm_interface) {
        m_arm_interface->AddMessageEvents();

        m_arm_interface->AddCommandReadState(StateTable, operating_state, "operating_state");

        m_arm_interface->AddCommandReadState(StateTable, measured_js, "measured_js");
        m_arm_interface->AddCommandReadState(StateTable, setpoint_js, "setpoint_js");
        m_arm_interface->AddCommandReadState(StateTable, measured_cp, "measured_cp");
        m_arm_interface->AddCommandReadState(StateTable, setpoint_cp, "setpoint_cp");
        m_arm_interface->AddCommandReadState(StateTable, measured_cv, "measured_cv");
        m_arm_interface->AddCommandReadState(StateTable, body_measured_cf, "body/measured_cf");
        m_arm_interface->AddCommandReadState(StateTable, body_external_cf, "external/measured_cf");
        m_arm_interface->AddCommandReadState(StateTable, gripper_measured_js, "gripper/measured_js");

        m_arm_interface->AddCommandReadState(StateTable, StateTable.PeriodStats, "period_statistics");

        m_arm_interface->AddCommandWrite(&mtsRemoteMTM::StateCommand, this, "state_command", std::string(""));
        m_arm_interface->AddCommandWrite(&mtsRemoteMTM::ServoCPVF, this, "servo_cpvf");
        m_arm_interface->AddCommandWrite(&mtsRemoteMTM::MoveCP, this, "move_cp");
        m_arm_interface->AddCommandWrite(&mtsRemoteMTM::LockOrientation, this, "lock_orientation");
        m_arm_interface->AddCommandVoid(&mtsRemoteMTM::UnlockOrientation, this, "unlock_orientation");
    }
}

void mtsRemoteMTM::Init(void)
{
    InitializeStateTable();
    InitializePubSub("remote/" + Name + "/");
    InitializeArmInterface();
}

void mtsRemoteMTM::Run(void)
{
    ProcessQueuedEvents();
    ProcessQueuedCommands();
}

void mtsRemoteMTM::StateCommand(const std::string & command)
{
    crtk_msgs::StringStamped msg;
    msg.string = command;

    state_command_pub.publish(msg);
}

void mtsRemoteMTM::ServoCPVF(const prmStateCartesian & cs)
{
    crtk_msgs::CartesianState msg;
    
    vctQuatRot3 quat(cs.Position().Rotation(), VCT_NORMALIZE);
    msg.Pose.orientation.x = quat.X();
    msg.Pose.orientation.y = quat.Y();
    msg.Pose.orientation.z = quat.Z();
    msg.Pose.orientation.w = quat.W();
    msg.Pose.position.x = cs.Position().Translation().X();
    msg.Pose.position.y = cs.Position().Translation().Y();
    msg.Pose.position.z = cs.Position().Translation().Z();
    msg.PoseIsDefined.data = cs.PositionIsDefined();

    msg.Twist.linear.x = cs.Velocity().Element(0);
    msg.Twist.linear.y = cs.Velocity().Element(1);
    msg.Twist.linear.z = cs.Velocity().Element(2);
    msg.Twist.angular.x = cs.Velocity().Element(3);
    msg.Twist.angular.y = cs.Velocity().Element(4);
    msg.Twist.angular.z = cs.Velocity().Element(5);
    msg.TwistIsDefined.data = cs.VelocityIsDefined();

    msg.Wrench.force.x = cs.Effort().Element(0);
    msg.Wrench.force.y = cs.Effort().Element(1);
    msg.Wrench.force.z = cs.Effort().Element(2);
    msg.Wrench.torque.x = cs.Effort().Element(3);
    msg.Wrench.torque.y = cs.Effort().Element(4);
    msg.Wrench.torque.z = cs.Effort().Element(5);
    msg.WrenchIsDefined.data = cs.EffortIsDefined();

    servo_cpvf_pub.publish(msg);
}

void mtsRemoteMTM::MoveCP(const prmPositionCartesianSet & move_cp)
{
    geometry_msgs::PoseStamped msg;

    vctQuatRot3 quat(move_cp.Goal().Rotation(), VCT_NORMALIZE);
    msg.pose.orientation.x = quat.X();
    msg.pose.orientation.y = quat.Y();
    msg.pose.orientation.z = quat.Z();
    msg.pose.orientation.w = quat.W();
    msg.pose.position.x = move_cp.Goal().Translation().X();
    msg.pose.position.y = move_cp.Goal().Translation().Y();
    msg.pose.position.z = move_cp.Goal().Translation().Z();

    move_cp_pub.publish(msg);
}

void mtsRemoteMTM::LockOrientation(const vctMatRot3 & orientation)
{
    geometry_msgs::Quaternion msg;

    vctQuatRot3 quaternion(orientation, VCT_NORMALIZE);
    msg.x = quaternion.X();
    msg.y = quaternion.Y();
    msg.z = quaternion.Z();
    msg.w = quaternion.W();

    lock_orientation_pub.publish(msg);
}

void mtsRemoteMTM::UnlockOrientation(void)
{
    std_msgs::Empty msg;
    unlock_orientation_pub.publish(msg);
}

void mtsRemoteMTM::OperatingStateCallback(const crtk_msgs::OperatingState& msg)
{
    try {
        operating_state.State() = prmOperatingState::StateTypeFromString(msg.state);
    } catch (...) {
        operating_state.State() = prmOperatingState::UNDEFINED;
    }
    operating_state.IsHomed() = msg.is_homed;
    operating_state.IsBusy() = msg.is_busy;
    operating_state.Valid() = true;
}

void mtsRemoteMTM::MeasuredJSCallback(const sensor_msgs::JointState& msg)
{
    measured_js.Position().SetSize(msg.position.size());
    std::copy(msg.position.begin(), msg.position.end(), measured_js.Position().begin());

    const auto velocity_size = msg.velocity.size();
    if (velocity_size > 0) {
        measured_js.Velocity().SetSize(velocity_size);
        std::copy(msg.velocity.begin(), msg.velocity.end(), measured_js.Velocity().begin());
    }

    const auto effort_size = msg.effort.size();
    if (effort_size > 0) {
        measured_js.Effort().SetSize(effort_size);
        std::copy(msg.effort.begin(), msg.effort.end(), measured_js.Effort().begin());
    }

    measured_js.Valid() = true;
}

void mtsRemoteMTM::SetpointJSCallback(const sensor_msgs::JointState& msg)
{
    setpoint_js.Position().SetSize(msg.position.size());
    std::copy(msg.position.begin(), msg.position.end(), setpoint_js.Position().begin());

    const auto velocity_size = msg.velocity.size();
    if (velocity_size > 0) {
        setpoint_js.Velocity().SetSize(velocity_size);
        std::copy(msg.velocity.begin(), msg.velocity.end(), setpoint_js.Velocity().begin());
    }

    const auto effort_size = msg.effort.size();
    if (effort_size > 0) {
        setpoint_js.Effort().SetSize(effort_size);
        std::copy(msg.effort.begin(), msg.effort.end(), setpoint_js.Effort().begin());
    }

    setpoint_js.Valid() = true;
}

void mtsRemoteMTM::GripperMeasuredJSCallback(const sensor_msgs::JointState& msg)
{
    gripper_measured_js.Position().SetSize(msg.position.size());
    std::copy(msg.position.begin(), msg.position.end(), gripper_measured_js.Position().begin());

    const auto velocity_size = msg.velocity.size();
    if (velocity_size > 0) {
        gripper_measured_js.Velocity().SetSize(velocity_size);
        std::copy(msg.velocity.begin(), msg.velocity.end(), gripper_measured_js.Velocity().begin());
    }

    const auto effort_size = msg.effort.size();
    if (effort_size > 0) {
        gripper_measured_js.Effort().SetSize(effort_size);
        std::copy(msg.effort.begin(), msg.effort.end(), gripper_measured_js.Effort().begin());
    }

    gripper_measured_js.Valid() = true;
}

void mtsRemoteMTM::MeasuredCPCallback(const geometry_msgs::PoseStamped& msg)
{
    measured_cp.Position().Translation().X() = msg.pose.position.x;
    measured_cp.Position().Translation().Y() = msg.pose.position.y;
    measured_cp.Position().Translation().Z() = msg.pose.position.z;
    vctQuatRot3 quaternion;
    quaternion.X() = msg.pose.orientation.x;
    quaternion.Y() = msg.pose.orientation.y;
    quaternion.Z() = msg.pose.orientation.z;
    quaternion.W() = msg.pose.orientation.w;
    vctMatRot3 rotation(quaternion, VCT_NORMALIZE);
    measured_cp.Position().Rotation().Assign(rotation);

    measured_cp.Valid() = true;
}

void mtsRemoteMTM::SetpointCPCallback(const geometry_msgs::PoseStamped& msg)
{
    setpoint_cp.Position().Translation().X() = msg.pose.position.x;
    setpoint_cp.Position().Translation().Y() = msg.pose.position.y;
    setpoint_cp.Position().Translation().Z() = msg.pose.position.z;
    vctQuatRot3 quaternion;
    quaternion.X() = msg.pose.orientation.x;
    quaternion.Y() = msg.pose.orientation.y;
    quaternion.Z() = msg.pose.orientation.z;
    quaternion.W() = msg.pose.orientation.w;
    vctMatRot3 rotation(quaternion, VCT_NORMALIZE);
    setpoint_cp.Position().Rotation().Assign(rotation);

    setpoint_cp.Valid() = true;
}

void mtsRemoteMTM::MeasuredCVCallback(const geometry_msgs::TwistStamped& msg)
{
    measured_cv.SetVelocity(vct6(msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                                 msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z));
    measured_cv.SetTimestamp(msg.header.stamp.toSec());

    measured_cv.Valid() = true;
}

void mtsRemoteMTM::MeasuredCFCallback(const geometry_msgs::WrenchStamped& msg)
{
    body_measured_cf.SetForce(vct6(msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                                   msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z));
    body_measured_cf.SetTimestamp(msg.header.stamp.toSec());

    body_measured_cf.Valid() = true;
}

void mtsRemoteMTM::ExternalCFCallback(const geometry_msgs::WrenchStamped& msg)
{
    body_external_cf.SetForce(vct6(msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                                   msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z));
    body_external_cf.SetTimestamp(msg.header.stamp.toSec());

    body_external_cf.Valid() = true;
}
