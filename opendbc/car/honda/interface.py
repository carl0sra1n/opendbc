#!/usr/bin/env python3
import numpy as np
from opendbc.car import create_button_events, get_safety_config, get_friction, structs, uds
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.disable_ecu import disable_ecu
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.values import CarControllerParams, HondaFlags, CAR, HONDA_BOSCH, HONDA_BOSCH_CANFD, \
                                                 HONDA_NIDEC_ALT_SCM_MESSAGES, HONDA_BOSCH_RADARLESS, HondaSafetyFlags
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.radar_interface import RadarInterface
from opendbc.car.interfaces import CarInterfaceBase, TorqueFromLateralAccelCallbackType, FRICTION_THRESHOLD, LatControlInputs

from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP, HondaSafetyFlagsSP

TransmissionType = structs.CarParams.TransmissionType


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    if CP.carFingerprint in HONDA_BOSCH:
      return CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX
    else:
      # NIDECs don't allow acceleration near cruise_speed,
      # so limit limits of pid to prevent windup
      ACCEL_MAX_VALS = [CarControllerParams.NIDEC_ACCEL_MAX, 0.2]
      ACCEL_MAX_BP = [cruise_speed - 2., cruise_speed - .2]
      return CarControllerParams.NIDEC_ACCEL_MIN, np.interp(current_speed, ACCEL_MAX_BP, ACCEL_MAX_VALS)

  def torque_from_lateral_accel_modded(self, latcontrol_inputs: LatControlInputs, torque_params: car.CarParams.LateralTorqueTuning, lateral_accel_error: float, lateral_accel_deadzone: float, friction_compensation: bool, gravity_adjusted: bool) -> float:
    threshold = 0.8
    threshold_lat_accel = 1/torque_params.latAccelFactor * threshold
    mod_factor = 2.0 # Lateral Accel
    # The default is a linear relationship between torque and lateral acceleration (accounting for road roll and steering friction)
    friction = get_friction(lateral_accel_error, lateral_accel_deadzone, FRICTION_THRESHOLD, torque_params, friction_compensation)
    if abs(latcontrol_inputs.lateral_acceleration) > threshold_lat_accel:
      modded_lat_accel_factor = float(torque_params.latAccelFactor) * mod_factor
      excess_lat_accel = abs(latcontrol_inputs.lateral_acceleration) - threshold_lat_accel
      torque = float(np.sign(latcontrol_inputs.lateral_acceleration)) * threshold_lat_accel / float(torque_params.latAccelFactor)
      torque += float(np.sign(latcontrol_inputs.lateral_acceleration)) * excess_lat_accel / modded_lat_accel_factor
    else:
      torque = latcontrol_inputs.lateral_acceleration / float(torque_params.latAccelFactor)
    return torque + friction

  def torque_from_lateral_accel(self) -> TorqueFromLateralAccelCallbackType:
    if not self.CP.enableGasInterceptorDEPRECATED:
      return self.torque_from_lateral_accel_modded
    else:
      return self.torque_from_lateral_accel_linear

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "honda"

    CAN = CanBus(ret, fingerprint)

    if candidate in HONDA_BOSCH:
      cfgs = [get_safety_config(structs.CarParams.SafetyModel.hondaBosch)]
      if candidate in HONDA_BOSCH_CANFD and CAN.pt >= 4:
        cfgs.insert(0, get_safety_config(structs.CarParams.SafetyModel.noOutput))
      ret.safetyConfigs = cfgs

      ret.radarUnavailable = True
      # Disable the radar and let openpilot control longitudinal
      # WARNING: THIS DISABLES AEB!
      # If Bosch radarless, this blocks ACC messages from the camera
      # TODO: get radar disable working on Bosch CANFD
      ret.alphaLongitudinalAvailable = candidate not in HONDA_BOSCH_CANFD
      ret.openpilotLongitudinalControl = alpha_long
      ret.pcmCruise = not ret.openpilotLongitudinalControl
    else:
      ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.hondaNidec)]
      ret.openpilotLongitudinalControl = True

      ret.pcmCruise = True

    if candidate == CAR.HONDA_CRV_5G:
      ret.enableBsm = 0x12f8bfa7 in fingerprint[CAN.radar]

    # Detect Bosch cars with new HUD msgs
    if any(0x33DA in f for f in fingerprint.values()):
      ret.flags |= HondaFlags.BOSCH_EXT_HUD.value

    if 0x184 in fingerprint[CAN.pt]:
      ret.flags |= HondaFlags.HYBRID.value

    if ret.flags & HondaFlags.ALLOW_MANUAL_TRANS and all(msg not in fingerprint[CAN.pt] for msg in (0x191, 0x1A3)):
      # Manual transmission support for allowlisted cars only, to prevent silent fall-through on auto-detection failures
      ret.transmissionType = TransmissionType.manual
    elif 0x191 in fingerprint[CAN.pt] and candidate != CAR.ACURA_RDX:
      # Traditional CVTs, gearshift position in GEARBOX_CVT
      ret.transmissionType = TransmissionType.cvt
    else:
      # Traditional autos, direct-drive EVs and eCVTs, gearshift position in GEARBOX_AUTO
      ret.transmissionType = TransmissionType.automatic

    ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0], [0]]
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
    ret.lateralTuning.pid.kf = 0.00006  # conservative feed-forward
    ret.steerActuatorDelay = 0.1

    if candidate in HONDA_BOSCH:
      ret.longitudinalActuatorDelay = 0.5 # s
      if candidate in HONDA_BOSCH_RADARLESS:
        ret.stopAccel = CarControllerParams.BOSCH_ACCEL_MIN  # stock uses -4.0 m/s^2 once stopped but limited by safety model
    else:
      # default longitudinal tuning for all hondas
      #tune.kiBP = [0.,  5.,   12.,  20.,  27.,  36.,  40.]
      #tune.kiV = [0.34, 0.234, 0.20, 0.17, 0.105, 0.09, 0.08]
      # toyota values noted above
      # ret.longitudinalTuning.kiBP = [0., 5., 35.]
      # ret.longitudinalTuning.kiV = [1.2, 0.8, 0.5]
      # honda values noted above
      ret.longitudinalTuning.kiBP = [0.,  5.,   12.,  20.,  27.,  36.]
      ret.longitudinalTuning.kiV = [0.4, 0.6, 0.8, 1.6, 1.8, 2.0]

    eps_modified = False
    for fw in car_fw:
      if fw.ecu == "eps" and b"," in fw.fwVersion:
        eps_modified = True

    if candidate == CAR.HONDA_CIVIC:
      if eps_modified:
      # Best practices for tuning modified Civic EPS firmware (written by Brett Pakkala aka Aragon).
      # As a general rule of thumb, beyond 2X~, larger values change the ramp-up and ramp-down rate, but not the maximum torque.
      # Therefore, if one is going above 2X~, it's best to start with 2X tuning values anyway and work your way down in increments of 10% until things feel smooth, if needed.
      #
      # The TorqueBP and TorqueV params work as follows: TorqueBP is the actual torque value listed in your EPS firmware file.
      # However, when sending a value to the car, the car only accepts values from 0 to 3840 — 0% being 0 and 100% being 3840. This is TorqueV.
      # We can mold these values to spread out the torque applied so that Openpilot understands that the relationship is not linear.
      #
      # Proportional (P), Integral (I), and Feed-forward (F) TUNING TIPS:
      # When tuning, the kp, ki, and kf should be changed at the same rate. For example, if you lower one value by 10%, lower all three by 10%.
      # If you alter TorqueBP in a certain way, ki/kp/kf should be altered in the opposite way. For example, if you divide TorqueBP by 2, multiply kp/ki/kf by 2.
      # Sometimes kf (feed-forward) can cause issues such as mild sway. It can be beneficial to try lowering this value separately from the rest, or setting it to 0 if nothing else works.
      #
      # Torque Controller TUNING TIPS:
      # The torque controller uses basic PIF params alongside the lateral acceleration factor and friction params.
      # Those PIF params can be found here: selfdrive/car/interfaces.py (line 267).
      # Lateral acceleration and friction params default to the ones found here: selfdrive/car/torque_data/params.toml
      # Since the torque controller expects a linear response, an additional function was created above in this file (line 39) to adjust the lateral acceleration factor once a certain threshold is crossed.
      # This makes it more in line with what a typical non-linear EPS firmware mod provides.
      # If manually tuning the lateral acceleration factor, note that lowering it will make Openpilot think you have less overall torque — thus turning earlier. Raising it will have the opposite effect.
      # Friction is a form of error correction. For very precise steering, turn friction very high — but this may cause many micro-corrections. For smoother response, lower friction. Adjust to your liking.
      #
      # LOW-PASS FILTER TUNING TIPS:
      # The low-pass filter located in selfdrive/car/honda/carcontroller.py (line 105) might create a delayed response depending on the other tuning params.
      # Feel free to revert it back to the stock version (listed below):
      # def rate_limit_steer(new_steer, last_steer):
      #   MAX_DELTA = 3 * DT_CTRL
      #   return clip(new_steer, last_steer - MAX_DELTA, last_steer + MAX_DELTA)
        ret.lateralParams.torqueBP = [0, 2560, 32767] # Max 16-bit torque.
        ret.lateralParams.torqueV = [0, 2560, 3840] # Value that gets sent to the EPS.
        ret.lateralTuning.pid.kf = 0.00003  # Modified feed-forward.
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.15], [0.05]] # Corresponding tuning
      else:
        ret.lateralTuning.pid.kf = 0.00006  # conservative feed-forward
        ret.lateralParams.torqueBP = [0x0, 0x917, 0xDC5, 0x1017, 0x119F, 0x140B, 0x1680, 0x6540, 0x8700]
        ret.lateralParams.torqueV = [0x0, 0x200, 0x300, 0x478, 0x5EC, 0x800, 0xA00, 0xE00, 0xF00]
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.3], [0.1]] # force modded values always on this fork

    elif candidate in (CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CIVIC_BOSCH_DIESEL):
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]
      if candidate == CAR.HONDA_CIVIC_BOSCH:
          CarControllerParams.BOSCH_GAS_LOOKUP_V = [0, 750]

    elif candidate == CAR.HONDA_CIVIC_2022:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kpV = [[0, 10], [0.05, 0.5]]
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kiV = [[0, 10], [0.0125, 0.125]]

    elif candidate == CAR.HONDA_ACCORD:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]

    elif candidate == CAR.ACURA_ILX:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate in (CAR.HONDA_CRV, CAR.HONDA_CRV_EU):
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 1000], [0, 1000]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.HONDA_CRV_5G:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.64], [0.192]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.HONDA_CRV_HYBRID:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]
      ret.wheelSpeedFactor = 1.025

    elif candidate == CAR.HONDA_FIT:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.05]]

    elif candidate == CAR.HONDA_FREED:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.05]]

    elif candidate in (CAR.HONDA_HRV, CAR.HONDA_HRV_3G):
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]
      if candidate == CAR.HONDA_HRV:
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.16], [0.025]]
        ret.wheelSpeedFactor = 1.025
      else:
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]  # TODO: can probably use some tuning

    elif candidate == CAR.ACURA_RDX:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 1000], [0, 1000]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.8], [0.24]]

    elif candidate == CAR.ACURA_RDX_3G:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 3840], [0, 3840]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.06]]

    elif candidate == CAR.HONDA_ODYSSEY:
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.28], [0.08]]
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end

    elif candidate == CAR.HONDA_PILOT:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.38], [0.11]]

    elif candidate == CAR.HONDA_RIDGELINE:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.38], [0.11]]

    elif candidate in (CAR.HONDA_INSIGHT, CAR.HONDA_NBOX_2G):
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]]

    elif candidate == CAR.HONDA_E:
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 4096], [0, 4096]]  # TODO: determine if there is a dead zone at the top end
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6], [0.18]] # TODO: can probably use some tuning

    elif candidate == CAR.HONDA_ODYSSEY_5G_MMR:
      # Stock camera sends up to 2560 during LKA operation and up to 3840 during RDM operation
      # Steer motor torque does rise a little above 2560, but not linearly, RDM also applies one-sided brake drag
      #ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560, 3072], [0, 2560, 3840]]
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560], [0, 2560]]
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)
      ret.steerActuatorDelay = 0.15
      CarControllerParams.BOSCH_GAS_LOOKUP_V = [0, 2000]
      if not ret.openpilotLongitudinalControl:
        # When using stock ACC, the radar intercepts and filters steering commands the EPS would otherwise accept
        ret.minSteerSpeed = 70. * CV.KPH_TO_MS

    # TODO-SP: remove when https://github.com/commaai/opendbc/pull/2687 is merged
    elif candidate == CAR.HONDA_CLARITY:
      pass

    else:
      ret.steerActuatorDelay = 0.15
      ret.lateralParams.torqueBP, ret.lateralParams.torqueV = [[0, 2560], [0, 2560]]
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    # These cars use alternate user brake msg (0x1BE)
    if 0x1BE in fingerprint[CAN.pt] and candidate in (CAR.HONDA_ACCORD, CAR.HONDA_HRV_3G, *HONDA_BOSCH_CANFD):
      ret.flags |= HondaFlags.BOSCH_ALT_BRAKE.value

    if ret.flags & HondaFlags.BOSCH_ALT_BRAKE:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.ALT_BRAKE.value
    if candidate in HONDA_NIDEC_ALT_SCM_MESSAGES:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.NIDEC_ALT.value
    if ret.openpilotLongitudinalControl and candidate in HONDA_BOSCH:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.BOSCH_LONG.value
    if candidate in HONDA_BOSCH_RADARLESS:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.RADARLESS.value
    if candidate in HONDA_BOSCH_CANFD:
      ret.safetyConfigs[-1].safetyParam |= HondaSafetyFlags.BOSCH_CANFD.value

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter. Otherwise, add 0.5 mph margin to not
    # conflict with PCM acc
    ret.autoResumeSng = candidate in (HONDA_BOSCH | {CAR.HONDA_CIVIC})
    ret.minEnableSpeed = -1. if ret.autoResumeSng else 25.51 * CV.MPH_TO_MS

    ret.steerLimitTimer = 0.8
    ret.radarDelay = 0.1

    return ret

  @staticmethod
  def _get_params_sp(stock_cp: structs.CarParams, ret: structs.CarParamsSP, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_sp: bool, docs: bool) -> structs.CarParamsSP:
    CAN = CanBus(stock_cp, fingerprint)

    for fw in car_fw:
      if fw.ecu == "eps" and b"," in fw.fwVersion:
        ret.flags |= HondaFlagsSP.EPS_MODIFIED.value
        stock_cp.dashcamOnly = False

    if candidate == CAR.HONDA_CIVIC:
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        # stock request input values:     0x0000, 0x00DE, 0x014D, 0x01EF, 0x0290, 0x0377, 0x0454, 0x0610, 0x06EE
        # stock request output values:    0x0000, 0x0917, 0x0DC5, 0x1017, 0x119F, 0x140B, 0x1680, 0x1680, 0x1680
        # modified request output values: 0x0000, 0x0917, 0x0DC5, 0x1017, 0x119F, 0x140B, 0x1680, 0x2880, 0x3180
        # stock filter output values:     0x009F, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108
        # modified filter output values:  0x009F, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0108, 0x0400, 0x0480
        # note: max request allowed is 4096, but request is capped at 3840 in firmware, so modifications result in 2x max
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2560, 8000], [0, 2560, 3840]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.1]]

    elif candidate in (CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CIVIC_BOSCH_DIESEL):
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2564, 8000], [0, 2564, 3840]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.09]]  # 2.5x Modded EPS

    elif candidate == CAR.HONDA_CIVIC_2022:
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2564, 8000], [0, 2564, 3840]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.09]]  # 2.5x Modded EPS

    elif candidate == CAR.HONDA_ACCORD:
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.09]]

    elif candidate == CAR.HONDA_CRV_5G:
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        # stock request input values:     0x0000, 0x00DB, 0x01BB, 0x0296, 0x0377, 0x0454, 0x0532, 0x0610, 0x067F
        # stock request output values:    0x0000, 0x0500, 0x0A15, 0x0E6D, 0x1100, 0x1200, 0x129A, 0x134D, 0x1400
        # modified request output values: 0x0000, 0x0500, 0x0A15, 0x0E6D, 0x1100, 0x1200, 0x1ACD, 0x239A, 0x2800
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2560, 10000], [0, 2560, 3840]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.21], [0.07]]

    elif candidate == CAR.HONDA_CLARITY:
      ret.safetyParam |= HondaSafetyFlagsSP.CLARITY
      stock_cp.autoResumeSng = True
      stock_cp.minEnableSpeed = -1
      if ret.flags & HondaFlagsSP.EPS_MODIFIED:
        for fw in car_fw:
          if fw.ecu == "eps" and b"-" not in fw.fwVersion and b"," in fw.fwVersion:
            stock_cp.lateralTuning.pid.kf = 0.00004
            stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 5760, 15360], [0, 2560, 3840]]
            stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.00525], [0.01725]]
          elif fw.ecu == "eps" and b"-" in fw.fwVersion and b"," in fw.fwVersion:
            stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 5760, 10240], [0, 2560, 3840]]
            stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.3], [0.1]]
      else:
        stock_cp.lateralParams.torqueBP, stock_cp.lateralParams.torqueV = [[0, 2560], [0, 2560]]
        stock_cp.lateralTuning.pid.kpV, stock_cp.lateralTuning.pid.kiV = [[0.8], [0.24]]

    if candidate in HONDA_BOSCH:
      pass
    else:
      ret.enableGasInterceptor = 0x201 in fingerprint[CAN.pt]
      stock_cp.pcmCruise = not ret.enableGasInterceptor

    if ret.enableGasInterceptor and candidate not in HONDA_BOSCH:
      ret.safetyParam |= HondaSafetyFlagsSP.GAS_INTERCEPTOR

    stock_cp.autoResumeSng = stock_cp.autoResumeSng or ret.enableGasInterceptor

    ret.intelligentCruiseButtonManagementAvailable = candidate in (HONDA_BOSCH - HONDA_BOSCH_CANFD) or \
                                                     (candidate in HONDA_BOSCH_CANFD and not is_release_sp)

    return ret

  @staticmethod
  def init(CP, CP_SP, can_recv, can_send, communication_control=None):
    if CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS) and CP.openpilotLongitudinalControl:
      # 0x80 silences response
      if communication_control is None:
        communication_control = bytes([uds.SERVICE_TYPE.COMMUNICATION_CONTROL, 0x80 | uds.CONTROL_TYPE.DISABLE_RX_DISABLE_TX,
                                       uds.MESSAGE_TYPE.NORMAL_AND_NETWORK_MANAGEMENT])
      disable_ecu(can_recv, can_send, bus=CanBus(CP).pt, addr=0x18DAB0F1, com_cont_req=communication_control)

  @staticmethod
  def deinit(CP, can_recv, can_send):
    communication_control = bytes([uds.SERVICE_TYPE.COMMUNICATION_CONTROL, 0x80 | uds.CONTROL_TYPE.ENABLE_RX_ENABLE_TX,
                                   uds.MESSAGE_TYPE.NORMAL_AND_NETWORK_MANAGEMENT])
    CarInterface.init(CP, can_recv, can_send, communication_control)
