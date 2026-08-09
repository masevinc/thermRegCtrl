// extract_wall_yplus.java
//
// STAR-CCM+ macro: mesh-quality check -- reads Wall Y+ (area-average and
// max) on the boundary/boundaries belonging to the manikin's own skin
// (name contains "dummy" but not "sensor", case-insensitive -- see below)
// in the currently loaded simulation, and writes one CSV row per boundary.
//
// Companion to extract_temp_csv.java / run_export_temp_csv.sh, same
// batch-per-sim-file pattern (getActiveSimulation(), not opening extra
// Simulation objects): run once per already-loaded .sim file, driven by
// run_export_wall_yplus.sh, which sets STARCCM_YPLUS_CSV_OUT per case
// before launching STAR-CCM+, e.g.:
//
//   STARCCM_YPLUS_CSV_OUT=/path/to/yplus_exports/coarse_min7_yplus.csv \
//     starccm+ -batch extract_wall_yplus.java /path/to/coarse_full_0805/save_stage3a_5min.sim
//
// If STARCCM_YPLUS_CSV_OUT is not set (e.g. running manually from the GUI),
// falls back to "<case_folder>_yplus.csv" next to the loaded .sim file.
//
// Wall Y+ field function: looked up by ID "WallYplus" (STAR-CCM+'s
// standard macro name for "Wall Y+"). If that's wrong for this version/
// setup, this macro prints every field function whose name contains "y"
// and "plus" instead of failing silently, so FIELD_FUNCTION_NAME can be
// fixed quickly. Same fallback behavior if BOUNDARY_NAME_FILTER matches
// nothing -- prints every boundary name it actually saw.
//
// Boundary naming (confirmed 2026-08-09 via a real batch-run's diagnostic
// output on case_debug_0804): the manikin's own skin is a SINGLE boundary,
// "Air.Dummy" -- NOT split per body region in this mesh (unlike the
// "DUMMY_FS_<region>" names seen in a different, older case report). The
// same sim also has 16 "...Sensor_XX_<region>.Sensor" boundaries -- these
// are the offset air-probe surfaces used for temperature/velocity
// monitors elsewhere in this pipeline, not solid walls, so Wall Y+ on them
// isn't meaningful -- excluded explicitly below (they also contain
// "dummy" in their full path name, so a plain substring match would wrongly
// grab them too). Net effect: this macro reports ONE whole-manikin Y+
// number per case (area-avg + max), not per-body-region -- good enough to
// answer "did this mesh achieve the target Y+ at all", but if per-region
// breakdown is needed later, "Air.Dummy" would need to be split into 16
// derived parts (e.g. Threshold-by-distance from each sensor point, same
// approach as the sensor-area-average velocity macro elsewhere in this repo).

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import star.common.*;
import star.base.report.AreaAverageReport;
import star.base.report.MaxReport;

public class extract_wall_yplus extends StarMacro {

  // Case-insensitive substring match on the boundary's presentation name.
  // Matches "Air.Dummy" (the manikin's own solid skin, what we want) while
  // excluding the 16 "...Sensor_XX_<region>.Sensor" offset-probe boundaries
  // (also contain "dummy" in their full path, but aren't solid walls) --
  // see the class-level comment above for how this was confirmed.
  private static final String BOUNDARY_NAME_FILTER = "dummy";
  private static final String BOUNDARY_NAME_EXCLUDE = "sensor";
  private static final String FIELD_FUNCTION_NAME = "WallYplus";

  public void execute() {

    Simulation sim = getActiveSimulation();

    FieldFunction yplusFF = null;
    try {
      yplusFF = sim.getFieldFunctionManager().getFunction(FIELD_FUNCTION_NAME);
    } catch (Exception e) {
      // fall through to the diagnostic branch below
    }
    if (yplusFF == null) {
      sim.println("ERROR: field function '" + FIELD_FUNCTION_NAME + "' not found. Candidates "
          + "(name contains 'y' and 'plus'):");
      for (FieldFunction ff : sim.getFieldFunctionManager().getObjects()) {
        String n = ff.getPresentationName().toLowerCase();
        if (n.contains("y") && n.contains("plus")) {
          sim.println("    id='" + ff.getFunctionName() + "' display='" + ff.getPresentationName() + "'");
        }
      }
      return;
    }

    List<Boundary> matched = new ArrayList<Boundary>();
    List<String> allBoundaryNames = new ArrayList<String>();
    for (Region region : sim.getRegionManager().getRegions()) {
      for (Boundary b : region.getBoundaryManager().getBoundaries()) {
        allBoundaryNames.add(region.getPresentationName() + "." + b.getPresentationName());
        String bNameLower = b.getPresentationName().toLowerCase();
        if (bNameLower.contains(BOUNDARY_NAME_FILTER) && !bNameLower.contains(BOUNDARY_NAME_EXCLUDE)) {
          matched.add(b);
        }
      }
    }

    if (matched.isEmpty()) {
      sim.println("ERROR: no boundary matched filter '" + BOUNDARY_NAME_FILTER + "'. All boundaries in this sim:");
      for (String n : allBoundaryNames) {
        sim.println("    " + n);
      }
      return;
    }

    String outPath = System.getenv("STARCCM_YPLUS_CSV_OUT");
    if (outPath == null || outPath.isEmpty()) {
      File simFile = new File(sim.getSessionPath());
      File caseDir = simFile.getParentFile();
      String caseName = (caseDir != null) ? caseDir.getName() : "case";
      outPath = new File(caseDir, caseName + "_yplus.csv").getPath();
    }
    File outFile = new File(outPath);
    if (outFile.getParentFile() != null) {
      outFile.getParentFile().mkdirs();
    }

    try {
      PrintWriter out = new PrintWriter(new FileWriter(outFile));
      out.println("boundary,area_avg_yplus,max_yplus");
      for (Boundary b : matched) {
        double avg = areaAverage(sim, yplusFF, b);
        double max = maxValue(sim, yplusFF, b);
        out.println(b.getPresentationName() + "," + avg + "," + max);
        sim.println(b.getPresentationName() + "  avg Y+=" + avg + "  max Y+=" + max);
      }
      out.close();
      sim.println("Exported Wall Y+ data to: " + outPath);
    } catch (Exception e) {
      sim.println("Failed to write CSV: " + e.getMessage());
    }
  }

  // Note: these reports are intentionally left in the ReportManager rather
  // than deleted -- Report has no destroy() method on this STAR-CCM+
  // version (20.06.010), and cleanup isn't needed anyway: each .sim file is
  // processed by its own `starccm+ -batch` invocation, which exits right
  // after this macro finishes, discarding everything.

  private double areaAverage(Simulation sim, FieldFunction ff, Boundary b) {
    AreaAverageReport rep = sim.getReportManager().createReport(AreaAverageReport.class);
    rep.setFieldFunction(ff);
    rep.getParts().setObjects(b);
    return rep.getReportMonitorValue();
  }

  private double maxValue(Simulation sim, FieldFunction ff, Boundary b) {
    MaxReport rep = sim.getReportManager().createReport(MaxReport.class);
    rep.setFieldFunction(ff);
    rep.getParts().setObjects(b);
    return rep.getReportMonitorValue();
  }
}
