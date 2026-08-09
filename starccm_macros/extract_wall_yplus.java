// extract_wall_yplus.java
//
// STAR-CCM+ macro: mesh-quality check -- reads Wall Y+ (area-average and
// max) on EVERY boundary in the currently loaded simulation (except the
// 16 offset air-probe "...Sensor_XX_<region>.Sensor" surfaces, which are
// not solid walls and would just add zero/meaningless rows -- see below),
// and writes one CSV row per boundary. Boundaries that aren't actual
// no-slip walls (inlets, outlets, symmetry planes, internal solid/solid
// interfaces, etc.) will simply come out with Y+ near/at 0 -- this macro
// does not try to guess/filter by boundary *type*, it reports everything
// and lets the numbers speak for themselves, since Alperen wants to see
// values on all surfaces where Y+ actually forms, not just the manikin.
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
// fixed quickly.
//
// Per-boundary computation is wrapped in try/catch: a handful of
// boundaries (confirmed 2026-08-09 on case_debug_0804: internal solid/
// solid interfaces like "Solid.internal-1", and possibly inlet/outlet/
// symmetry types) may not support this field function at all and throw
// rather than just returning 0 -- those are logged as "ERROR" in the CSV
// (not silently dropped, and not allowed to kill the whole batch run).

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import star.common.*;
import star.base.report.AreaAverageReport;
import star.base.report.MaxReport;

public class extract_wall_yplus extends StarMacro {

  // Only real exclusion: the 16 offset air-probe boundaries (not solid
  // walls -- see class-level comment). Everything else is reported.
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
    for (Region region : sim.getRegionManager().getRegions()) {
      for (Boundary b : region.getBoundaryManager().getBoundaries()) {
        String bNameLower = b.getPresentationName().toLowerCase();
        if (!bNameLower.contains(BOUNDARY_NAME_EXCLUDE)) {
          matched.add(b);
        }
      }
    }

    if (matched.isEmpty()) {
      sim.println("ERROR: no boundaries found at all (unexpected).");
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
      out.println("region,boundary,area_avg_yplus,max_yplus");
      for (Boundary b : matched) {
        String regionName = b.getRegion().getPresentationName();
        String row;
        try {
          double avg = areaAverage(sim, yplusFF, b);
          double max = maxValue(sim, yplusFF, b);
          row = regionName + "," + b.getPresentationName() + "," + avg + "," + max;
          sim.println(regionName + "." + b.getPresentationName() + "  avg Y+=" + avg + "  max Y+=" + max);
        } catch (Exception e) {
          row = regionName + "," + b.getPresentationName() + ",ERROR,ERROR";
          sim.println(regionName + "." + b.getPresentationName() + "  ERROR: " + e.getMessage());
        }
        out.println(row);
      }
      out.close();
      sim.println("Exported Wall Y+ data (" + matched.size() + " boundaries) to: " + outPath);
    } catch (Exception e) {
      sim.println("Failed to write CSV: " + e.getMessage());
    }
  }

  // Note: reports are intentionally left in the ReportManager rather than
  // deleted -- Report has no destroy() method on this STAR-CCM+ version
  // (20.06.010), and cleanup isn't needed anyway: each .sim file is
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
