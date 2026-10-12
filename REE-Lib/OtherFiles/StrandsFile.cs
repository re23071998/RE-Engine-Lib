using System.Numerics;
using System.Runtime.InteropServices;
using ReeLib.Common;

namespace ReeLib
{
    /// <summary>
    /// One point of the "pos" section (16 bytes, see <see cref="StrandsLod.PosStride"/>).
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct StrandsPoint
    {
        public Vector3 Position;

        /// <summary>u16 strand radius. RE4R convention: real radius = radius / 100000 (unverified on RE9).</summary>
        public ushort Radius;

        /// <summary>u8 along-strand parameter 0..255; drives the root-to-tip colour ramp.</summary>
        public byte CurvePos;

        /// <summary>u8 per-strand constant; feeds the shader's per-strand colour randomisation.</summary>
        public byte Flag;
    }

    /// <summary>
    /// Typed view on one u32 of the "curve" section. There is one entry per point except the
    /// last point of each strand (a 1-point strand has no entry at all).
    /// </summary>
    public readonly struct StrandsCurveEntry
    {
        public const uint IndexMask = 0x0FFFFFFF;
        public const int FlagShift = 28;

        public readonly uint Value;

        public StrandsCurveEntry(uint value) => Value = value;

        /// <summary>Point index (28 bits).</summary>
        public uint Index => Value & IndexMask;

        /// <summary>
        /// Bit flags: bit0 = strand root, bit1 = second-to-last point (both set = 2-point strand).
        /// The last point of a strand carries no entry.
        /// </summary>
        public uint Flags => Value >> FlagShift;

        public bool IsRoot => (Flags & 1) != 0;
        public bool IsSecondToLast => (Flags & 2) != 0;

        public static uint Encode(uint index, uint flags) => (index & IndexMask) | (flags << FlagShift);

        public override string ToString() => $"{Index} (flags {Flags})";
    }

    /// <summary>
    /// One LOD of a strand hair file: the five per-LOD sections pos / curve / root / point / guide.
    /// </summary>
    public class StrandsLod
    {
        /// <summary>Bytes per point in the "pos" section.</summary>
        public const int PosStride = 16;

        /// <summary>Bytes per point in the "guide" section (all versions).</summary>
        public const int GuideStride = 24;

        public const int GuideLaneCount = 12;

        // Guide lane indices, identical for the AoS (.20) and SoA (.31/.33) layouts.
        // Semantics (RE9/RE4R/pragmata decode, re-strand-hair-exp SPEC.md §4.5):
        //   curve0..2 = bound guide strand ids into the same file, per strand (0xFFFF = unused slot)
        //   point0..2 = point along the guide curve, per point
        //   w0..w2    = barycentric guide weights (half floats, sum 1.0), per point
        //   w3..w5    = half floats, only partly decoded (w3: 0 at every root, ~1 at interior points)
        public const int LaneCurve0 = 0;
        public const int LaneCurve1 = 1;
        public const int LaneCurve2 = 2;
        public const int LanePoint0 = 3;
        public const int LanePoint1 = 4;
        public const int LanePoint2 = 5;
        public const int LaneW0 = 6;
        public const int LaneW1 = 7;
        public const int LaneW2 = 8;
        public const int LaneW3 = 9;
        public const int LaneW4 = 10;
        public const int LaneW5 = 11;

        /// <summary>0xFFFF marks an unused guide slot (curve1/curve2/point2 on RE9/pragmata files).</summary>
        public const ushort UnusedGuideSlot = 0xFFFF;

        public StrandsPoint[] Points = [];

        /// <summary>Raw curve-section entries; decode with <see cref="StrandsCurveEntry"/>.</summary>
        public uint[] Curves = [];

        /// <summary>First point index of every strand; strand length = Roots[i+1] - Roots[i].</summary>
        public uint[] Roots = [];

        /// <summary>Strand index of every point.</summary>
        public uint[] PointStrands = [];

        /// <summary>
        /// Guide block as 12 parallel u16 lanes of <see cref="PointCount"/> values each.
        /// Weight lanes are IEEE half-float bit patterns (see <see cref="BitsToHalf"/>).
        /// </summary>
        public ushort[][] GuideLanes = [[], [], [], [], [], [], [], [], [], [], [], []];

        public int PointCount => Points.Length;
        public int StrandCount => Roots.Length;

        public static Half BitsToHalf(ushort bits) => BitConverter.Int16BitsToHalf((short)bits);
        public static ushort HalfToBits(Half value) => BitConverter.HalfToUInt16Bits(value);

        internal void ReadGuide(FileHandler handler, bool interleaved)
        {
            int n = Points.Length;
            for (int lane = 0; lane < GuideLaneCount; lane++)
                GuideLanes[lane] = new ushort[n];

            if (interleaved)
            {
                // RE4 (.20): AoS, 24 B per point: u16 curve[3], u16 point[3], half w[3], half w2[3].
                var raw = handler.ReadArray<ushort>(n * GuideLaneCount);
                for (int p = 0; p < n; p++)
                    for (int lane = 0; lane < GuideLaneCount; lane++)
                        GuideLanes[lane][p] = raw[p * GuideLaneCount + lane];
            }
            else
            {
                // RE9 (.33) / pragmata (.31): SoA, six 4-byte sub-arrays, each holding
                // (u16 lane 2k, u16 lane 2k+1) per point. The SoA layout is load-bearing:
                // an AoS-interleaved block parses but renders nothing in-game.
                for (int k = 0; k < GuideLaneCount / 2; k++)
                {
                    var arr = handler.ReadArray<uint>(n);
                    for (int p = 0; p < n; p++)
                    {
                        GuideLanes[2 * k][p] = (ushort)arr[p];
                        GuideLanes[2 * k + 1][p] = (ushort)(arr[p] >> 16);
                    }
                }
            }
        }

        internal void WriteGuide(FileHandler handler, bool interleaved)
        {
            int n = Points.Length;
            for (int lane = 0; lane < GuideLaneCount; lane++)
            {
                if (GuideLanes[lane].Length != n)
                    throw new InvalidOperationException($"GuideLanes[{lane}].Length {GuideLanes[lane].Length} != point count {n}");
            }

            if (interleaved)
            {
                var raw = new ushort[n * GuideLaneCount];
                for (int p = 0; p < n; p++)
                    for (int lane = 0; lane < GuideLaneCount; lane++)
                        raw[p * GuideLaneCount + lane] = GuideLanes[lane][p];
                handler.WriteArray(raw);
            }
            else
            {
                for (int k = 0; k < GuideLaneCount / 2; k++)
                {
                    var arr = new uint[n];
                    for (int p = 0; p < n; p++)
                        arr[p] = GuideLanes[2 * k][p] | ((uint)GuideLanes[2 * k + 1][p] << 16);
                    handler.WriteArray(arr);
                }
            }
        }
    }

    /// <summary>
    /// Strand hair file, magic "STRD" (<c>*.strands.NN</c>).<br/>
    /// Version 20 = RE4 (188-byte header, 2 LODs, AoS guide block).<br/>
    /// Version 31 = pragmata (244-byte header = the version-33 header truncated after the cull
    /// box, asset constant moved to 0xF0).<br/>
    /// Version 33 = RE9 (292-byte header, 4 LODs, SoA guide block, "perStrand" trailing section,
    /// five extra trailing sections on <c>*_guide.strands</c> files).<br/>
    /// Per-LOD section order: pos (16 B/point), curve (4 B/curve entry), root (4 B/strand),
    /// point (4 B/point), guide (24 B/point). Trailing: uv (8 B/strand), perStrand (4 B/strand,
    /// .31/.33 only), then g0..g4 (RE9 guide files only).
    /// <para>
    /// All size/count header fields are recomputed from the parsed arrays on write; the g54/g64
    /// markers are recomputed as the point prefix sums of the first PrimaryGuideCount /
    /// GuideCount strands (shipped files store exactly these sums; a zeroed marker renders
    /// nothing). Engine notes (verified in-game, re-strand-hair-exp SPEC.md §7): all 4 LOD
    /// sections must be present with non-zero sizes, the guide block must keep the documented
    /// layout, and both bounding boxes must be recomputed for new geometry — a zeroed cull box
    /// renders nothing.
    /// </para>
    /// <para>
    /// Byte-exact read/write round-trip verified offline against 15 shipped samples
    /// (12x RE9 .33 incl. 4 _guide files, 2x RE4 .20, 1x pragmata .31).
    /// </para>
    /// </summary>
    public class StrandsFile(FileHandler fileHandler) : BaseFile(fileHandler)
    {
        public const uint Magic = 0x44525453; // 'STRD'

        public const int VersionRE4 = 20;
        public const int VersionPragmata = 31;
        public const int VersionRE9 = 33;

        /// <summary>
        /// File version (20 = RE4, 31 = pragmata, 33 = RE9), captured from the numeric suffix
        /// of the input handler's file name at construction/read time, so that writing through
        /// a different handler (WriteTo to a stream) keeps the parsed version.
        /// </summary>
        public int Version => version != 0 ? version : FileHandler.FileVersion;

        private int version = fileHandler.FileVersion;

        public int LodCount => Version switch
        {
            VersionRE4 => 2,
            VersionPragmata or VersionRE9 => 4,
            _ => 0,
        };

        /// <summary>
        /// Header 0x78 ("count2"): number of primary guide strands. The first PrimaryGuideCount
        /// strands are self-referencing guides (curve0 == own strand index).
        /// </summary>
        public uint PrimaryGuideCount;

        /// <summary>
        /// Header 0x7C ("N"): total guide strand count — the first N strands are the guides
        /// (primaries, then secondaries that bind only to primaries). 0 = single tier.
        /// Not present in RE4 files.
        /// </summary>
        public uint GuideCount;

        /// <summary>Header 0x80: LOD0 AABB, stored (maxX, maxY, maxZ) then (minX, minY, minZ).</summary>
        public Vector3 BboxMax;
        public Vector3 BboxMin;

        /// <summary>Width triple (avg, max, min) — shading/tint control, not a geometric width.</summary>
        public Vector3 Width;

        /// <summary>Header 0xD8 expanded cull box (min first), roughly 2x the AABB, offset slightly +Y.</summary>
        public Vector3 CullBoxMin;
        public Vector3 CullBoxMax;

        /// <summary>Header 0x120 (RE9) / 0xF0 (pragmata): per-asset constant, undecoded.</summary>
        public uint AssetConstant;

        public StrandsLod[] Lods = [];

        /// <summary>"uv" trailing section: root UV (u, v) per strand.</summary>
        public float[] Uv = [];

        /// <summary>
        /// "perStrand" trailing section (.31/.33 only): u32 per strand, 0..99 on RE9 render
        /// files, mixed values and 0xFFFFFFFF on guide files; purpose undecoded.
        /// </summary>
        public uint[] PerStrand = [];

        /// <summary>
        /// RE9 <c>*_guide.strands</c> only: the five extra trailing sections g0..g4, sized by the
        /// header fields at 0xA8..0xB8. Content undecoded, preserved verbatim.
        /// </summary>
        public byte[][] GuideTails = [[], [], [], [], []];

        /// <summary>Bytes past the documented layout, preserved verbatim (empty on all shipped samples).</summary>
        public byte[] RawTrailing = [];

        public int StrandCount => Lods.Length > 0 ? Lods[0].StrandCount : 0;

        private static void ReadValues(FileHandler handler, uint[] values)
        {
            for (int i = 0; i < values.Length; i++)
                values[i] = handler.Read<uint>();
        }

        private void CheckVersion()
        {
            if (LodCount == 0)
                throw new NotSupportedException(
                    $"Unsupported .strands version {Version} (expected {VersionRE4}/{VersionPragmata}/{VersionRE9}). " +
                    "FileHandler.FileVersion is parsed from the numeric suffix of the file name, e.g. \".strands.33\".");
        }

        /// <summary>
        /// g54/g64 header marker for one LOD: total point count of the file's first
        /// <paramref name="strandCount"/> strands, i.e. the root index of strand `strandCount`.
        /// Shipped files store exactly this prefix sum; the write path recomputes it from
        /// <see cref="StrandsLod.Roots"/>.
        /// </summary>
        public uint GuideMarker(int lod, uint strandCount)
        {
            var roots = Lods[lod].Roots;
            if (strandCount == 0) return 0;
            if (strandCount >= (uint)roots.Length) return (uint)Lods[lod].PointStrands.Length;
            return roots[strandCount];
        }

        protected override bool DoRead()
        {
            var handler = FileHandler;
            version = handler.FileVersion;
            CheckVersion();
            int v = Version;
            int lodCount = LodCount;
            bool isRe4 = v == VersionRE4;

            if (handler.Read<uint>() != Magic)
                throw new Exception("Invalid STRD file");

            var curveCounts = new uint[lodCount];
            var posSizes = new uint[lodCount];
            var curveSizes = new uint[lodCount];
            var rootSizes = new uint[lodCount];
            var pointSizes = new uint[lodCount];
            var g54 = new uint[lodCount];
            var g64 = new uint[lodCount];
            var tailSizes = new uint[5];

            if (isRe4)
            {
                // 188-byte header; every undocumented gap is zero in all shipped samples.
                handler.ReadNull(8);                    // 0x04
                ReadValues(handler, curveCounts);       // 0x0C curveCount[2]
                handler.ReadNull(8);                    // 0x14
                ReadValues(handler, posSizes);          // 0x1C posSize[2]
                handler.ReadNull(8);                    // 0x24
                ReadValues(handler, curveSizes);        // 0x2C curveSize[2]
                handler.ReadNull(8);                    // 0x34
                ReadValues(handler, rootSizes);         // 0x3C rootSize[2]
                handler.ReadNull(8);                    // 0x44
                ReadValues(handler, pointSizes);        // 0x4C pointSize[2]
                handler.ReadNull(8);                    // 0x54
                ReadValues(handler, g54);               // 0x5C g54[2]
            }
            else
            {
                ReadValues(handler, curveCounts);       // 0x04 curveCount[4]
                ReadValues(handler, posSizes);          // 0x14 posSize[4]
                ReadValues(handler, curveSizes);        // 0x24 curveSize[4]
                ReadValues(handler, rootSizes);         // 0x34 rootSize[4]
                ReadValues(handler, pointSizes);        // 0x44 pointSize[4]
                ReadValues(handler, g54);               // 0x54 g54[4]
                ReadValues(handler, g64);               // 0x64 g64[4]
            }

            var uvSize = handler.Read<uint>();
            PrimaryGuideCount = handler.Read<uint>();
            if (!isRe4)
                GuideCount = handler.Read<uint>();
            BboxMax = handler.Read<Vector3>();
            BboxMin = handler.Read<Vector3>();

            if (isRe4)
            {
                handler.ReadNull(8);                    // 0x84
                var guideSizes = new uint[lodCount];
                ReadValues(handler, guideSizes);        // 0x8C guideSize[2]
                handler.ReadNull(28);                   // 0x94
                for (int i = 0; i < lodCount; i++)
                {
                    if (guideSizes[i] != StrandsLod.GuideStride * (posSizes[i] / StrandsLod.PosStride))
                        throw new Exception($"Invalid .strands.20: guideSize[{i}] = {guideSizes[i]}, expected 24 x pointCount");
                }
            }
            else
            {
                var pointSizesDup = new uint[lodCount]; // 0x98 pointSize[4] duplicate
                ReadValues(handler, pointSizesDup);
                for (int i = 0; i < lodCount; i++)
                {
                    if (pointSizesDup[i] != pointSizes[i])
                        Log.Warn($"pointSize[{i}] duplicate in header (0x98) = {pointSizesDup[i]} != {pointSizes[i]}, rewriting with the recomputed value");
                }
                for (int i = 0; i < tailSizes.Length; i++)
                    tailSizes[i] = handler.Read<uint>(); // 0xA8 g0..g4 sizes
                handler.ReadNull(12);                   // 0xBC
            }

            Width = handler.Read<Vector3>();            // 0xB0 (.20) / 0xC8 (.31/.33)

            if (!isRe4)
            {
                var rootSize0Dup = handler.Read<uint>(); // 0xD4 rootSize[0] duplicate
                if (rootSize0Dup != rootSizes[0])
                    Log.Warn($"rootSize[0] duplicate in header (0xD4) = {rootSize0Dup} != {rootSizes[0]}, rewriting with the recomputed value");
                CullBoxMin = handler.Read<Vector3>();   // 0xD8
                CullBoxMax = handler.Read<Vector3>();   // 0xE4
                if (v == VersionRE9)
                    handler.ReadNull(48);               // 0xF0 reserved
                AssetConstant = handler.Read<uint>();   // 0xF0 (.31) / 0x120 (.33)
            }

            Lods = new StrandsLod[lodCount];
            for (int i = 0; i < lodCount; i++)
            {
                var lod = new StrandsLod();
                lod.Points = handler.ReadArray<StrandsPoint>((int)(posSizes[i] / StrandsLod.PosStride));
                lod.Curves = handler.ReadArray<uint>((int)(curveSizes[i] / 4));
                lod.Roots = handler.ReadArray<uint>((int)(rootSizes[i] / 4));
                lod.PointStrands = handler.ReadArray<uint>((int)(pointSizes[i] / 4));
                lod.ReadGuide(handler, isRe4);
                Lods[i] = lod;
            }

            if (uvSize % 4 != 0)
                throw new Exception($"Invalid .strands: uvSize {uvSize} not a multiple of 4");
            Uv = handler.ReadArray<float>((int)(uvSize / 4));
            if (uvSize != (uint)StrandCount * 8)
                Log.Warn($"uvSize {uvSize} != 8 x strandCount {StrandCount}");

            if (!isRe4)
            {
                PerStrand = handler.ReadArray<uint>(StrandCount);
                for (int i = 0; i < GuideTails.Length; i++)
                    GuideTails[i] = tailSizes[i] > 0 ? handler.ReadBytes((int)tailSizes[i]) : [];
            }

            var remaining = handler.Stream.Length - handler.Stream.Position;
            if (remaining > 0)
            {
                RawTrailing = handler.ReadBytes((int)remaining);
                Log.Warn($"{remaining} unaccounted trailing bytes, preserved verbatim");
            }

            // g54/g64 are stored prefix sums; warn when a file does not match the identity
            // (the write path always recomputes them).
            for (int i = 0; i < lodCount; i++)
            {
                if (g54[i] != GuideMarker(i, PrimaryGuideCount))
                    Log.Warn($"g54[{i}] = {g54[i]} != point prefix sum of the first {PrimaryGuideCount} strands ({GuideMarker(i, PrimaryGuideCount)})");
                if (!isRe4 && g64[i] != GuideMarker(i, GuideCount))
                    Log.Warn($"g64[{i}] = {g64[i]} != point prefix sum of the first {GuideCount} strands ({GuideMarker(i, GuideCount)})");
            }
            return true;
        }

        protected override bool DoWrite()
        {
            CheckVersion();
            var handler = FileHandler;
            int v = Version;
            int lodCount = LodCount;
            bool isRe4 = v == VersionRE4;

            if (Lods.Length != lodCount)
                throw new InvalidOperationException($"StrandsFile: {Lods.Length} LODs, expected {lodCount} for version {v}");
            if (Uv.Length != StrandCount * 2)
                Log.Warn($"Uv length {Uv.Length} != 2 x strand count {StrandCount * 2}");

            handler.Write(Magic);
            if (isRe4)
            {
                handler.WriteNull(8);
                for (int i = 0; i < lodCount; i++) handler.Write((uint)Lods[i].Curves.Length);
                handler.WriteNull(8);
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].Points.Length * StrandsLod.PosStride));
                handler.WriteNull(8);
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].Curves.Length * 4));
                handler.WriteNull(8);
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].Roots.Length * 4));
                handler.WriteNull(8);
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].PointStrands.Length * 4));
                handler.WriteNull(8);
                for (int i = 0; i < lodCount; i++) handler.Write(GuideMarker(i, PrimaryGuideCount));
                handler.Write((uint)(Uv.Length * 4));       // uvSize = 8 x strandCount
                handler.Write(PrimaryGuideCount);
                handler.Write(BboxMax);
                handler.Write(BboxMin);
                handler.WriteNull(8);
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].Points.Length * StrandsLod.GuideStride)); // guideSize = 24 x points
                handler.WriteNull(28);
                handler.Write(Width);
            }
            else
            {
                for (int i = 0; i < lodCount; i++) handler.Write((uint)Lods[i].Curves.Length);
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].Points.Length * StrandsLod.PosStride));
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].Curves.Length * 4));
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].Roots.Length * 4));
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].PointStrands.Length * 4));
                for (int i = 0; i < lodCount; i++) handler.Write(GuideMarker(i, PrimaryGuideCount));
                for (int i = 0; i < lodCount; i++) handler.Write(GuideMarker(i, GuideCount));
                handler.Write((uint)(Uv.Length * 4));
                handler.Write(PrimaryGuideCount);
                handler.Write(GuideCount);
                handler.Write(BboxMax);
                handler.Write(BboxMin);
                for (int i = 0; i < lodCount; i++) handler.Write((uint)(Lods[i].PointStrands.Length * 4)); // pointSize duplicate
                for (int i = 0; i < GuideTails.Length; i++) handler.Write((uint)GuideTails[i].Length);
                handler.WriteNull(12);
                handler.Write(Width);
                handler.Write((uint)(Lods[0].Roots.Length * 4)); // rootSize[0] duplicate
                handler.Write(CullBoxMin);
                handler.Write(CullBoxMax);
                if (v == VersionRE9)
                    handler.WriteNull(48);
                handler.Write(AssetConstant);
            }

            for (int i = 0; i < lodCount; i++)
            {
                var lod = Lods[i];
                handler.WriteArray(lod.Points);
                handler.WriteArray(lod.Curves);
                handler.WriteArray(lod.Roots);
                handler.WriteArray(lod.PointStrands);
                lod.WriteGuide(handler, isRe4);
            }
            handler.WriteArray(Uv);
            if (!isRe4)
            {
                if (PerStrand.Length != StrandCount)
                    Log.Warn($"PerStrand length {PerStrand.Length} != strand count {StrandCount}");
                handler.WriteArray(PerStrand);
                foreach (var tail in GuideTails)
                {
                    if (tail.Length > 0)
                        handler.WriteBytes(tail);
                }
            }
            if (RawTrailing.Length > 0)
                handler.WriteBytes(RawTrailing);
            return true;
        }

        /// <summary>
        /// Recomputes the header bounding box from the LOD0 point positions. Call after editing
        /// geometry. The expanded cull box is NOT touched here — recompute it as well for new
        /// hair, a zeroed cull box renders nothing.
        /// </summary>
        public void RecomputeBounds()
        {
            if (Lods.Length == 0) return;
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            foreach (var p in Lods[0].Points)
            {
                min = Vector3.Min(min, p.Position);
                max = Vector3.Max(max, p.Position);
            }
            BboxMin = min;
            BboxMax = max;
        }

        public override string ToString() => $"StrandsFile v{Version}, {LodCount} LODs, {StrandCount} strands";
    }
}
